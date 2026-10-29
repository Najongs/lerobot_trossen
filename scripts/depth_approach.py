#!/usr/bin/env python
"""cam_high depth 로 「앞의 목표까지 거리」를 재며 베이스를 조금씩 전진시킨다 — depth 거리 ↔ 오도메트리 대조.

목표: 앞에 보이는 물체(튜브 랙)를 잡으러 다가갈 때 depth 를 거리 센서로 쓸 수 있는지 본다.

원리: 베이스가 회전 없이 직진하면 고정된 물체의 카메라 좌표 P 는 이동한 만큼 반대로 평행이동한다.
  |P_k − P_0| = 실제 이동 거리 (카메라 기울기와 무관)  → 오도메트리와 비교
  f = −(P_k − P_0)/|…|                               → 카메라 좌표계에서 본 베이스 전진 방향(= 카메라 기울기)
  P·f                                                → 목표까지 「앞쪽 수평 거리」
물체는 다가갈수록 화면에서 커지고 위로 올라가므로, 정지할 때마다 다중 배율 템플릿 매칭으로 다시 찾는다.
화면 밖으로 나가면(매칭 점수 낮음) 그 정지에서 그 물체는 빠진다.

순서: 정지 0 측정 → +step 전진 · 측정 … (합 --forward) → 시작점으로 후진 · 측정(왕복 일치 확인).
정지마다 프레임 --frames 장의 픽셀별 중앙값 depth 를 쓴다.

  uv run python scripts/depth_approach.py --dry-run        # 베이스 가짜(무접촉), 카메라는 실제 — 흐름 확인
  uv run python scripts/depth_approach.py                  # 🚨 베이스가 실제로 움직인다

🚨 dry-run 이 아니면 베이스가 앞으로 --forward(기본 0.45 m, 상한 0.5 m) 갔다가 돌아온다.
   전진 전마다 Enter 확인을 받는다. 진행 방향을 비우고 비상정지에 손이 닿는 곳에서 실행할 것.
   정지마다 화면 중앙 아래 영역의 가까운 depth(5 퍼센타일)가 --min-clearance 보다 가까우면 더 가지 않는다.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from depth_probe import SERIALS, colorbar, colorize  # noqa: E402
from measure_base import _check_alive, _connect, _drive_to  # noqa: E402

# 랙 기준 이미지 — 10/27 PC2 cam_high, 랙 검은 몸통(x 314~358, y 78~103) + 여백. 시작 자세가 바뀌어도
# 정지마다 이 템플릿으로 랙을 새로 찾는다 (고정 좌표 ROI 는 자세가 바뀌면 엉뚱한 곳을 잡았다 — 10/27 211837 런)
RACK_REF = ("outputs/depth_probe/20261027_205657/cam_high_color.png", (310, 74, 362, 107))
MATCH_MIN = 0.55  # 정규화 상관 점수 이하면 「못 찾음」
DARK_MAX = 70     # 랙 몸통 픽셀 = 회색조 이 값 미만 (흰 튜브·책상·벽 제외)


def open_cam(pc: str, width: int, height: int, fps: int):
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    import pyrealsense2 as rs

    cam = RealSenseCamera(RealSenseCameraConfig(
        serial_number_or_name=SERIALS[pc]["cam_high"], fps=fps, width=width, height=height, use_depth=True))
    cam.connect()
    scale = cam.rs_profile.get_device().first_depth_sensor().get_depth_scale()
    intr = cam.rs_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    return cam, scale, (intr.fx, intr.fy, intr.ppx, intr.ppy)


def capture(cam, n: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """n 프레임 → (마지막 color RGB, 픽셀별 중앙값 depth m — 유효 프레임만, 절반 이상 무효면 0)."""
    colors, depths = [], []
    for _ in range(n):
        cam.async_read(timeout_ms=1000)
        with cam.frame_lock:
            colors.append(cam.latest_color_frame.copy())
            depths.append(cam.latest_depth_frame.astype(np.float32) * scale)
    d = np.stack(depths)
    d[d <= 0] = np.nan
    valid_frac = np.isfinite(d).mean(axis=0)
    med = np.nanmedian(np.where(np.isfinite(d), d, np.nan), axis=0) if n > 1 else d[0]
    med = np.where(valid_frac >= 0.5, med, 0).astype(np.float32)
    return colors[-1], np.nan_to_num(med)


def find(gray: np.ndarray, tmpl: np.ndarray, scales) -> tuple[float, tuple[int, int, int, int]]:
    best = (-1.0, (0, 0, 0, 0))
    for s in scales:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        if t.shape[0] >= gray.shape[0] or t.shape[1] >= gray.shape[1] or min(t.shape) < 8:
            continue
        r = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(r)
        if score > best[0]:
            best = (float(score), (x, y, x + t.shape[1], y + t.shape[0]))
    return best


def roi_point(depth: np.ndarray, box, K, gray: np.ndarray | None = None) -> tuple[np.ndarray | None, float]:
    """박스 안 유효 depth 픽셀을 3D 로 펴서 중앙값 점과 유효 비율. gray 를 주면 어두운(DARK_MAX 미만) 픽셀만."""
    fx, fy, cx, cy = K
    x0, y0, x1, y1 = box
    d = depth[max(y0, 0):y1, max(x0, 0):x1]
    if d.size == 0:
        return None, 0.0
    mask = d > 0
    if gray is not None:
        mask &= gray[max(y0, 0):y1, max(x0, 0):x1] < DARK_MAX
    ys, xs = np.nonzero(mask)
    frac = len(ys) / d.size
    if len(ys) < 20:
        return None, frac
    z = d[ys, xs]
    # 앞쪽 물체만 — 박스 안에 걸린 뒷배경(벽 등)을 빼려고 가장 가까운 절반 + 5 cm 만 쓴다
    keep = z <= np.percentile(z, 50) + 0.05
    u, v, z = xs[keep] + max(x0, 0), ys[keep] + max(y0, 0), z[keep]
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
    return np.median(P, axis=0), frac


def clearance(depth: np.ndarray) -> float:
    """화면 가운데 1/2 폭·전체 높이의 1 퍼센타일 depth — 바닥·다리뿐 아니라 책상 앞 모서리도 본다."""
    h, w = depth.shape
    d = depth[:, w // 4: 3 * w // 4]
    d = d[d > 0]
    return float(np.percentile(d, 1)) if d.size else float("nan")


def odom_x(base, x0y0th0):
    base.update_state()
    x, y, _ = base.get_pose()
    x0, y0, th0 = x0y0th0
    return (x - x0) * math.cos(th0) + (y - y0) * math.sin(th0)


def save_stop(out: Path, k: int, color, depth, boxes):
    bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    vis = colorize(depth, 0.5, 2.5, 0.5)
    for name, (score, box, P) in boxes.items():
        c = (0, 255, 0) if P is not None else (0, 0, 255)
        for img in (bgr, vis):
            cv2.rectangle(img, box[:2], box[2:], c, 2)
            label = f"{name} {np.linalg.norm(P):.3f}m" if P is not None else f"{name} ?"
            cv2.putText(img, label, (box[0], max(box[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    bar = colorbar(vis.shape[0], 0.5, 2.5, 0.5)
    cv2.imwrite(str(out / f"stop{k}_color.png"), bgr)
    cv2.imwrite(str(out / f"stop{k}_depth.png"), np.hstack([vis, bar]))
    np.save(out / f"stop{k}_depth_m.npy", depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="베이스를 가짜로 (카메라는 실제로 연다)")
    ap.add_argument("--pc", default=os.environ.get("TROSSEN_PC", "pc1"), choices=list(SERIALS))
    ap.add_argument("--forward", type=float, default=0.45, help="총 이동 거리 m — 음수면 후진 (절댓값 ≤ --max-travel)")
    ap.add_argument("--max-travel", type=float, default=0.5, help="이동 거리 절댓값 상한 m (10/27 사용자 지정 0.5, 그 뒤 요청 시만 넓힌다)")
    ap.add_argument("--step", type=float, default=0.125, help="정지 간격 m")
    ap.add_argument("--speed", type=float, default=0.08, help="직진 속도 m/s")
    ap.add_argument("--frames", type=int, default=15, help="정지마다 받을 프레임 수")
    ap.add_argument("--min-clearance", type=float, default=0.45,
                    help="정지 때 화면 가운데 영역의 가장 가까운 depth 가 이보다 작으면 더 전진하지 않는다 (m)")
    ap.add_argument("--roi", action="append", default=[],
                    help="추가 추적 물체 name=x0,y0,x1,y1 (정지 0 화면 640×480 픽셀). 랙은 항상 기준 이미지로 찾는다")
    ap.add_argument("--no-return", action="store_true", help="마지막 전진 위치에 멈춘다 (줄자 실측용)")
    ap.add_argument("--yes", action="store_true", help="전진 전 Enter 확인 생략 (dry-run 에서만 허용)")
    ap.add_argument("--out", default="outputs/depth_approach")
    args = ap.parse_args()

    if abs(args.forward) > args.max_travel + 1e-9:
        raise SystemExit(f"--forward 절댓값이 --max-travel {args.max_travel} m 를 넘는다")
    if args.yes and not args.dry_run:
        raise SystemExit("--yes 는 dry-run 에서만")
    rois = {}
    if args.roi:
        for r in args.roi:
            n, v = r.split("=")
            rois[n] = tuple(int(x) for x in v.split(","))

    out = Path(args.out) / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dry" if args.dry_run else ""))
    out.mkdir(parents=True, exist_ok=True)
    n_steps = max(1, round(abs(args.forward) / args.step))
    targets = [args.forward * (i + 1) / n_steps for i in range(n_steps)]

    cam, scale, K = open_cam(args.pc, 640, 480, 30)
    base, cleanup = _connect(args.dry_run)
    stops, tmpls = [], {}
    try:
        # 오도 0 점 = 시작 확인 동작 전 (줄자 기준점과 같게)
        base.update_state()
        origin = tuple(base.get_pose())
        if not args.dry_run:
            _check_alive(base, 0, probe=0.05)  # 5 cm/s · 0.6 s — 약 3 cm 앞으로 갔다 멈춘다
        scales = np.arange(0.5, 3.0, 0.05)
        ref = cv2.imread(str(_REPO_ROOT / RACK_REF[0]), cv2.IMREAD_GRAYSCALE)
        rx0, ry0, rx1, ry1 = RACK_REF[1]
        rack_tmpl = ref[ry0:ry1, rx0:rx1].copy()

        def measure(k: int, label: str):
            color, depth = capture(cam, args.frames, scale)
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            boxes = {}
            score, found = find(gray, rack_tmpl, scales)
            P, _ = roi_point(depth, found, K, gray) if score >= MATCH_MIN else (None, 0.0)
            boxes["rack"] = (score, found, P)
            for name, box in rois.items():
                if k == 0:
                    x0, y0, x1, y1 = box
                    tmpls[name] = gray[y0:y1, x0:x1].copy()
                    score, found = 1.0, box
                else:
                    score, found = find(gray, tmpls[name], scales)
                P, frac = roi_point(depth, found, K) if score >= MATCH_MIN else (None, 0.0)
                boxes[name] = (score, found, P)
            ox = odom_x(base, origin)
            clr = clearance(depth)
            save_stop(out, k, color, depth, boxes)
            stops.append({"stop": k, "label": label, "odom_m": ox, "clearance_m": clr,
                          "rois": {n: {"score": s, "box": list(b), "P": None if P is None else P.tolist()}
                                   for n, (s, b, P) in boxes.items()}})
            seen = " · ".join(f"{n} {np.linalg.norm(P):.3f}m(점수 {s:.2f})" if P is not None else f"{n} 못 찾음({s:.2f})"
                              for n, (s, b, P) in boxes.items())
            print(f"[정지 {k} {label}] 오도 {ox:+.3f} m · 앞 최소 depth {clr:.2f} m · {seen}")
            return clr

        clr = measure(0, "시작")
        for i, t in enumerate(targets, 1):
            if args.forward > 0 and not (clr >= args.min_clearance):  # 후진은 cam_high 로 뒤를 못 본다
                print(f"앞 최소 depth {clr:.2f} m < {args.min_clearance} — 더 전진하지 않는다")
                break
            if not args.yes:
                d = t - (targets[i - 2] if i > 1 else 0)
                input(f"\n  {d:+.2f} m {'전진' if d > 0 else '후진'} (누적 {t:+.2f} m) — 진행 방향 비었으면 Enter: ")
            _drive_to(base, 0, t - odom_x(base, origin), args.speed, 0.05, 15)
            clr = measure(i, f"{'전진' if t > 0 else '후진'} {t:+.2f}")
        if not args.no_return:
            back = -odom_x(base, origin)
            if not args.yes:
                input(f"\n  시작점으로 후진 {back:+.2f} m — 뒤 비었으면 Enter: ")
            _drive_to(base, 0, back, args.speed, 0.05, 20)
            measure(len(stops), "복귀")
    finally:
        cleanup()
        cam.disconnect()

    # 분석 — 정지 0 대비 각 물체의 평행이동
    s0 = stops[0]
    rows, fwd = [], []
    for s in stops[1:]:
        dodo = s["odom_m"] - s0["odom_m"]
        for n, r in s["rois"].items():
            P0, P = s0["rois"][n]["P"], r["P"]
            if P0 is None or P is None:
                continue
            dP = np.array(P) - np.array(P0)
            rows.append({"stop": s["stop"], "roi": n, "odom_m": dodo, "depth_shift_m": float(np.linalg.norm(dP)),
                         "dP": dP.tolist()})
            # 실제로 움직인 정지만 (dry-run 은 걸러짐) · 오매칭(이동량이 오도와 40% 넘게 다름)은 방향 추정에서 뺀다
            if abs(dodo) > 0.1 and abs(np.linalg.norm(dP) / abs(dodo) - 1) < 0.4:
                fwd.append(-dP / np.linalg.norm(dP) * np.sign(dodo))
    report = {"args": vars(args), "dry_run": args.dry_run, "K": K, "depth_scale": scale, "stops": stops, "shifts": rows}
    print("\n  정지   물체        오도(m)   depth 이동(m)   차(cm)")
    for r in rows:
        print(f"  {r['stop']:>3}   {r['roi']:10s} {r['odom_m']:+.3f}    {r['depth_shift_m']:.3f}        "
              f"{(r['depth_shift_m'] - abs(r['odom_m'])) * 100:+.1f}")
    if fwd:
        f = np.median(np.stack(fwd), axis=0)
        f /= np.linalg.norm(f)
        pitch = math.degrees(math.atan2(-f[1], f[2]))
        report["forward_dir_cam"] = f.tolist()
        report["camera_pitch_down_deg"] = pitch
        print(f"\n  카메라 좌표의 전진 방향 f = {np.round(f, 3).tolist()} → 카메라가 아래로 {pitch:.1f}° 숙임")
        for s in stops:
            P = s["rois"].get("rack", {}).get("P")
            if P is not None:
                print(f"  정지 {s['stop']} 오도 {s['odom_m']:+.3f} m · 랙까지 앞쪽 수평 거리 {float(np.dot(P, f)):.3f} m")
    if args.dry_run:
        print("\n  (dry-run: 베이스는 가짜라 카메라 장면이 그대로 — 이동량 비교는 의미 없다. 흐름 확인용)")
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
