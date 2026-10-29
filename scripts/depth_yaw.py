#!/usr/bin/env python
"""cam_high RGB-D 두 장 사이의 베이스 움직임(yaw·수평 이동)을 잰다 — 각도기·줄자 대신.

방식(RGB-D 비주얼 오도메트리): 컬러에서 ORB 특징점을 잡아 두 장 사이를 대응시키고, 각 점의 depth 로 3D 점을 만든 뒤
RANSAC + Kabsch 로 강체 변환 P0 ≈ R·Pk + t 를 맞춘다. 수평 기준은 바닥 평면(RANSAC) — 카메라 숙임을 몰라도 된다.
  yaw  = R 을 바닥 법선 축으로 본 회전 (+ = 왼쪽/CCW). **회전각은 depth 비율 오차(×0.945)와 무관하다**
  이동 = t 의 수평 성분 (카메라 원점이 움직인 양 — 회전 중심이 카메라와 다르면 제자리 회전에도 호를 그린다).
         depth 비율 오차를 그대로 받는다

10/27 검증 (outputs/depth_rotate/20261027_220528 · depth_approach/2026102721*):
  제자리 ±30° 회전 — VO 와 오도가 1~4° 안. 직진 0.5~0.6 m — VO yaw 0±1.5°, 이동은 오도를 따라감.
  처음 시도한 「세로 면 법선 방위각 히스토그램」 은 depth 노이즈가 만든 카메라 고정 법선(방위 0°)에 끌려
  회전을 늘 ≈0 으로 냈다 — 쓰지 마라.

  uv run python scripts/depth_yaw.py --run outputs/depth_rotate/<런>     # 정지 0 대비 각 정지 + 이웃 정지끼리
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# PC2 cam_high depth 내부 파라미터 (10/27 depth_probe 실측, 640×480)
K_PC2_HIGH = (389.3585205078125, 389.3585205078125, 321.6883850097656, 239.26876831054688)


def points(depth: np.ndarray, K, step: int = 2) -> np.ndarray:
    """depth(m) → 카메라 좌표 점 격자 (H/step, W/step, 3). 무효는 NaN."""
    fx, fy, cx, cy = K
    d = depth[::step, ::step].astype(np.float64)
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    P = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], -1)
    P[d <= 0] = np.nan
    return P


def floor_normal(P: np.ndarray, iters: int = 300, thresh: float = 0.015, seed: int = 0) -> np.ndarray:
    """화면 아래 40 % 의 점으로 바닥 평면 RANSAC. 카메라 쪽(위)을 향한 단위 법선."""
    h = P.shape[0]
    Q = P[int(h * 0.6):].reshape(-1, 3)
    Q = Q[np.isfinite(Q).all(1)]
    if len(Q) < 100:
        raise RuntimeError("바닥 점이 부족하다")
    rng = np.random.default_rng(seed)
    if len(Q) > 4000:  # 3만 점 × 300회면 2.7 s — 표본 4천이면 0.05 s, 법선은 0.1° 안에서 같다
        Q = Q[rng.choice(len(Q), 4000, replace=False)]
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        cnt = int((np.abs((Q - a) @ n) < thresh).sum())
        if cnt > best_n:
            best_n, best = cnt, (n, a)
    n, a = best
    inl = Q[np.abs((Q - a) @ n) < thresh]
    c = inl.mean(0)
    n = np.linalg.svd(inl - c)[2][-1]  # 최소제곱으로 다듬기
    if n @ (-c) < 0:  # 카메라(원점) 쪽을 향하게
        n = -n
    return n


def level_basis(up: np.ndarray):
    """up 에 수직인 앞(카메라 z 를 바닥에 투영)·왼쪽 축."""
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    return fwd, left


def _point3d(kp, depth, K, max_z: float):
    fx, fy, cx, cy = K
    u, v = int(round(kp.pt[0])), int(round(kp.pt[1]))
    w = depth[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3]
    w = w[w > 0]
    if w.size < 5:
        return None
    z = float(np.median(w))
    if z > max_z:
        return None
    return np.array([(kp.pt[0] - cx) / fx * z, (kp.pt[1] - cy) / fy * z, z])


def _kabsch(A, B):
    """A ≈ R·B + t 인 R, t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, ca - R @ cb


def motion(gray0, depth0, gray1, depth1, K=K_PC2_HIGH, max_z: float = 2.5, thr: float = 0.04,
           iters: int = 2000, seed: int = 0, nfeatures: int = 4000) -> dict:
    """정지 0 → 1 사이 카메라(=베이스) 움직임. gray 는 uint8 회색조, depth 는 m."""
    orb = cv2.ORB_create(nfeatures)
    kp0, des0 = orb.detectAndCompute(gray0, None)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    A, B = [], []
    if des0 is not None and des1 is not None:
        for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des0, des1, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                p0 = _point3d(kp0[pair[0].queryIdx], depth0, K, max_z)
                p1 = _point3d(kp1[pair[0].trainIdx], depth1, K, max_z)
                if p0 is not None and p1 is not None:
                    A.append(p0)
                    B.append(p1)
    if len(A) < 6:
        return {"yaw_deg": None, "matches": len(A), "inliers": 0}
    A, B = np.array(A), np.array(B)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(A), 3, replace=False)
        R, t = _kabsch(A[idx], B[idx])
        inl = np.linalg.norm(A - (B @ R.T + t), axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    R, t = _kabsch(A[best], B[best])
    up = floor_normal(points(depth0, K))
    f, l = level_basis(up)
    Rf = R @ f
    th = t - up * (t @ up)
    return {
        "yaw_deg": float(np.degrees(np.arctan2(Rf @ l, Rf @ f))),
        "move_m": float(np.linalg.norm(th)), "fwd_m": float(t @ f), "left_m": float(t @ l),
        "matches": len(A), "inliers": int(best.sum()),
    }


def _load(run: Path, k: int):
    g = cv2.cvtColor(cv2.imread(str(run / f"stop{k}_color.png")), cv2.COLOR_BGR2GRAY)
    return g, np.load(run / f"stop{k}_depth_m.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="depth_rotate/depth_approach 런 디렉터리")
    args = ap.parse_args()
    run = Path(args.run)
    rep = json.loads((run / "report.json").read_text())
    st = {s["stop"]: s for s in rep["stops"]}
    ks = sorted(st)

    def odo(a, b):
        if "odom_yaw_deg" in st[b]:
            return f"오도 Δyaw {st[b]['odom_yaw_deg'] - st[a]['odom_yaw_deg']:+7.2f}°"
        return f"오도 Δx {st[b]['odom_m'] - st[a]['odom_m']:+.3f} m"

    for title, pairs in (("정지 0 대비", [(ks[0], k) for k in ks[1:]]),
                         ("이웃 정지끼리", list(zip(ks[:-1], ks[1:])))):
        print(f"── {title}")
        for a, b in pairs:
            r = motion(*_load(run, a), *_load(run, b))
            if r["yaw_deg"] is None:
                print(f"  {a}→{b}: 매칭 부족 ({r['matches']})")
                continue
            print(f"  {a}→{b}: {odo(a, b)} | VO yaw {r['yaw_deg']:+7.2f}° · 수평 이동 {r['move_m']:.3f} m "
                  f"(앞 {r['fwd_m']:+.3f} · 왼 {r['left_m']:+.3f}) · 인라이어 {r['inliers']}/{r['matches']}")


if __name__ == "__main__":
    main()


def pose_vs_ref(ref_gray, cur_gray, cur_depth, K=K_PC2_HIGH, max_z: float = 3.0, nfeatures: int = 4000) -> dict:
    """depth 없는 기준 영상(예: 녹화 첫 프레임) 대비 현재 카메라 자세 — 2D(기준) ↔ 3D(현재, depth) 대응을 PnP RANSAC 으로.

    돌려주는 yaw/이동은 「현재 → 기준」 으로 가려면 베이스가 얼마나 돌고 움직여야 하는지 (+ 왼쪽, 바닥 평면 기준 수평).
    녹화 데이터셋에는 depth 가 없어서 motion() 을 못 쓴다 — 재생 시작 자세 맞추기용 (10/28)."""
    fx, fy, cx, cy = K
    orb = cv2.ORB_create(nfeatures)
    kr, dr = orb.detectAndCompute(ref_gray, None)
    kc, dc = orb.detectAndCompute(cur_gray, None)
    obj, img = [], []
    if dr is not None and dc is not None:
        for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dc, dr, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                p3 = _point3d(kc[pair[0].queryIdx], cur_depth, K, max_z)
                if p3 is not None:
                    obj.append(p3)
                    img.append(kr[pair[0].trainIdx].pt)
    if len(obj) < 8:
        return {"yaw_deg": None, "matches": len(obj), "inliers": 0}
    obj, img = np.array(obj, np.float64), np.array(img, np.float64)
    Km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, Km, None, reprojectionError=3.0, iterationsCount=2000,
                                             flags=cv2.SOLVEPNP_SQPNP)
    if not ok or inl is None or len(inl) < 8:
        return {"yaw_deg": None, "matches": len(obj), "inliers": 0 if inl is None else len(inl)}
    R, _ = cv2.Rodrigues(rvec)  # P_ref = R·P_cur + t
    # 기준 카메라 자세를 현재 카메라 좌표로: 기준 원점 = −Rᵀt, 기준 z 축 = Rᵀ·(0,0,1)
    Rt = R.T
    ref_origin = -Rt @ tvec.ravel()
    up = floor_normal(points(cur_depth, K))
    f, l = level_basis(up)
    zr = Rt @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.degrees(np.arctan2(zr @ l, zr @ f)))  # 기준 카메라가 현재보다 왼쪽을 보면 +
    return {"yaw_deg": yaw, "fwd_m": float(ref_origin @ f), "left_m": float(ref_origin @ l),
            "matches": len(obj), "inliers": int(len(inl))}
