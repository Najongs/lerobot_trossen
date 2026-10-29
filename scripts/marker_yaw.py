#!/usr/bin/env python
"""cam_high 에 보이는 ArUco/AprilTag 마커·체커보드로 베이스 yaw 를 잰다 — VO(depth_yaw.py)보다 정밀한 외부 기준.

회전은 PnP 로 푼다. 마커·체커보드의 실제 크기를 몰라도 **회전은 정확**하다(크기는 이동량에만 들어간다). 평면 PnP 는 해가
둘 나오는데(IPPE 플립 모호성) depth 로 잰 각 점의 거리 비와 맞는 쪽을 고른다 — 이때 맞춰진 배율이 곧 실제 한 변 길이라
정지마다 같은 값이 나오는지가 자체 검증이 된다.
yaw = 바닥 평면(depth_yaw.floor_normal, 런 평균) 기준 수평면에서 마커 x 축(윗변 0→1 · 체커보드 행 방향)의 방위각.
베이스가 왼쪽(+)으로 돌면 카메라에서 본 마커는 오른쪽으로 도니 yaw 변화 = −(방위각 변화). 체커보드는 행 방향이 180° 모호해
Δ 를 (−90, 90] 으로 접는다. 보드가 바닥에 평평하든 기울여 세워져 있든 상관없다.

버린 방식 (10/27 첫 판): 「마커가 바닥 평면 위에 있다」고 보고 모서리 시선과 바닥의 교점으로 3D 를 만들었다 — 기울여 세운
보드에서 −167°·+89° 같은 값이 나왔고, 작은 마커(한 변 42 px) 안쪽 depth 로 맞춘 평면은 정지 중에도 ±1.7° 흔들렸다.

  uv run python scripts/marker_yaw.py --live                       # cam_high 를 열어 검출·방위를 1 초 간격으로 출력 (설치 확인)
  uv run python scripts/marker_yaw.py --run outputs/depth_rotate/<런>   # 저장된 정지들에서 이웃 정지끼리 yaw 변화 (오도·VO 와 나란히)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_yaw import K_PC2_HIGH, _load, floor_normal, level_basis, motion, points  # noqa: E402

# PC2 cam_high color 내부 파라미터 (10/27 depth_probe 실측) — 모서리는 color 에서 잡으므로 PnP 는 이걸 쓴다
K_PC2_HIGH_COLOR = (392.6899719238281, 392.1737976074219, 319.078857421875, 241.7447509765625)
DICTS = ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50", "DICT_ARUCO_ORIGINAL",
         "DICT_APRILTAG_36h11", "DICT_APRILTAG_25h9", "DICT_APRILTAG_16h5"]
CHESS = (8, 6)  # 10/27 PC2 체커보드 내부 코너 (가로, 세로)
SQUARE = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], float)  # IPPE_SQUARE 순서
_DETECTORS = {}


def detect(gray: np.ndarray):
    """모든 사전에서 찾은 마커 [(사전, id, 4×2 모서리)]. 같은 자리 중복은 먼저 찾은 것."""
    found = []
    for name in DICTS:
        if name not in _DETECTORS:
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            _DETECTORS[name] = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), params)
        corners, ids, _ = _DETECTORS[name].detectMarkers(gray)
        for c, i in zip(corners or [], [] if ids is None else ids.ravel()):
            c = c.reshape(4, 2)
            if all(np.linalg.norm(c.mean(0) - f[2].mean(0)) > 10 for f in found):
                found.append((name, int(i), c))
    return found


def _depth_at(depth, uv, r: int = 2) -> float:
    u, v = int(round(uv[0])), int(round(uv[1]))
    w = depth[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
    w = w[w > 0]
    return float(np.median(w)) if w.size >= 3 else float("nan")


def pnp_plane(obj: np.ndarray, img: np.ndarray, depth: np.ndarray, K=K_PC2_HIGH_COLOR, square: bool = False):
    """z=0 평면 위 점들의 PnP. 두 해 중 depth 거리 비와 맞는 쪽 → (R, 배율 m/단위, 거리 잔차 m). 못 풀면 None."""
    fx, fy, cx, cy = K
    Km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj.astype(np.float64), img.astype(np.float64).reshape(-1, 1, 2), Km, None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE if square else cv2.SOLVEPNP_IPPE)
    if not n:
        return None
    z = np.array([_depth_at(depth, p) for p in img])
    good = np.isfinite(z)
    if good.sum() < 3:
        return None
    best = None
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        pz = (obj @ R.T + tv.ravel())[:, 2][good]
        s = float((pz @ z[good]) / (pz @ pz))  # 배율: 단위 → m
        res = float(np.sqrt(np.mean((s * pz - z[good]) ** 2)))
        if best is None or res < best[3]:
            best = (R, tv.ravel(), s, res)
    return best


def floor_up(depth, K=K_PC2_HIGH) -> np.ndarray:
    return floor_normal(points(depth, K))


def measure(gray, depth, up, K=K_PC2_HIGH_COLOR) -> dict:
    """정지 한 장의 마커별 방위각. up = 바닥 법선(런 평균을 주는 것이 좋다)."""
    f, l = level_basis(up)

    def entry(R, t, s, res, px, mod):
        x, nrm = R[:, 0], R[:, 2]
        return {"az_deg": math.degrees(math.atan2(x @ l, x @ f)), "px": float(px), "unit_m": s, "res_m": res,
                "tilt_deg": math.degrees(math.acos(min(1.0, abs(float(nrm @ up))))), "mod": mod,
                "center_m": (t * s).tolist()}  # 마커 중심 (카메라 좌표, m) — PnP 이동 × 배율

    out = {}
    # 같은 ID 가 여럿이면(10/27: 같은 마커를 두 장 인쇄) 눕힘(flat)/세움(stand)으로 가른다 — 하나만 보여도 이름이 같다
    for name, i, c in detect(gray):
        r = pnp_plane(SQUARE, c, depth, K, square=True)
        if r is None:
            continue
        e = entry(*r, np.linalg.norm(c[1] - c[0]), 360)
        key = f"{name}:{i}:{'flat' if e['tilt_deg'] < 45 else 'stand'}"
        while key in out:
            key += "'"
        out[key] = e
    ok, cc = cv2.findChessboardCornersSB(gray, CHESS, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if ok:
        cc = cc.reshape(-1, 2)
        obj = np.array([[j, -i, 0] for i in range(CHESS[1]) for j in range(CHESS[0])], float)
        r = pnp_plane(obj, cc, depth, K)
        if r is not None:
            out["chess"] = entry(*r, np.linalg.norm(cc[1] - cc[0]), 180)
    # 마커가 둘 이상이면 중심끼리 잇는 선의 방위각 — 세운 마커도 위치는 정확하니 여기에 쓴다 (윗변 방향은 정면에서 못 정한다)
    ks = sorted(k for k in out if k != "chess")
    for a_i in range(len(ks)):
        for b_i in range(a_i + 1, len(ks)):
            v = np.array(out[ks[b_i]]["center_m"]) - np.array(out[ks[a_i]]["center_m"])
            out[f"pair({ks[a_i].split(':')[-1]}→{ks[b_i].split(':')[-1]})"] = {
                "az_deg": math.degrees(math.atan2(v @ l, v @ f)), "px": 0.0, "unit_m": float(np.linalg.norm(v)),
                "res_m": 0.0, "tilt_deg": 0.0, "mod": 360}
    return out


def wrap(a: float, mod: int) -> float:
    return (a + mod / 2) % mod - mod / 2


def fmt(k: str, v: dict) -> str:
    if k.startswith("pair"):
        return f"{k} 방위 {v['az_deg']:+.2f}° · 거리 {v['unit_m'] * 100:.1f} cm"
    return (f"{k} 방위 {v['az_deg']:+.2f}° · 한 칸 {v['unit_m'] * 100:.1f} cm · 기울기 {v['tilt_deg']:.0f}° "
            f"· 잔차 {v['res_m'] * 100:.1f} cm · {v['px']:.0f}px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--pc", default=os.environ.get("TROSSEN_PC", "pc1"))
    args = ap.parse_args()

    if args.live:
        from depth_approach import capture, open_cam
        cam, scale, Kd = open_cam(args.pc, 640, 480, 30)
        try:
            for _ in range(5):
                color, depth = capture(cam, 10, scale)
                m = measure(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), depth, floor_up(depth, Kd))
                print("마커 없음" if not m else " | ".join(fmt(k, v) for k, v in m.items()))
                time.sleep(1)
        finally:
            cam.disconnect()
        return

    run = Path(args.run)
    st = {s["stop"]: s for s in json.loads((run / "report.json").read_text())["stops"]}
    ks = sorted(st)
    frames = {k: _load(run, k) for k in ks}
    ups = np.array([floor_up(d) for _, d in frames.values()])
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    print(f"바닥 법선 흔들림(정지별, 평균 대비) {np.degrees(np.arccos(np.clip(ups @ up, -1, 1))).round(2).tolist()}°")
    ms = {k: measure(g, d, up) for k, (g, d) in frames.items()}
    for k in ks:
        print(f"[정지 {k}] " + (" | ".join(fmt(n, v) for n, v in ms[k].items()) or "검출 없음"))
    print("\n  구간    오도 Δyaw    VO Δyaw   마커 Δyaw")
    for a, b in zip(ks[:-1], ks[1:]):
        do = st[b]["odom_yaw_deg"] - st[a]["odom_yaw_deg"]
        vo = motion(*frames[a], *frames[b])["yaw_deg"]
        parts = [f"{n.split(':')[-1] if ':' in n and not n.startswith('pair') else n} {-wrap(ms[b][n]['az_deg'] - ms[a][n]['az_deg'], ms[a][n]['mod']):+.2f}"
                 for n in sorted(set(ms[a]) & set(ms[b]))]
        print(f"  {a}→{b}   {do:+8.2f}°   {('—' if vo is None else f'{vo:+7.2f}°')}   {' · '.join(parts) or '공통 검출 없음'}")


if __name__ == "__main__":
    main()
