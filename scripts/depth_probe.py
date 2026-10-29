#!/usr/bin/env python
"""RealSense 3대 depth 점검 — 카메라만 연다(팔·베이스 무접촉).

lerobot 의 RealSenseCamera(use_depth=True) 경로를 그대로 쓴다 — 나중에 관측에 depth 를 넣을 때 탈 경로라서.
카메라별로 N 프레임을 받아:
  - 실제 fps (color+depth 동시, 3대 동시 — USB 대역폭 확인)
  - depth_scale (lerobot docstring 은 「mm」 라지만 변환 없이 z16 원값을 준다 — 장치 값으로 확인)
  - 유효 픽셀 비율 · 거리 분포(m) · 시간 노이즈(픽셀별 프레임 간 표준편차의 중앙값)
  - depth→color extrinsic (같은 해상도로 받는데 정렬이 맞는지)
를 재고, 마지막 프레임의 color / 컬러맵 depth / 겹친 그림 PNG 와 stats.json 을 저장한다.

  uv run python scripts/depth_probe.py                 # TROSSEN_PC 로 시리얼 선택
  uv run python scripts/depth_probe.py --frames 90 --out outputs/depth_probe
  uv run python scripts/depth_probe.py --render outputs/depth_probe/<런>   # 카메라 없이 다시 그리기
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

# scripts/robot_hw.sh 와 같은 값
SERIALS = {
    "pc1": {"cam_high": "230422273501", "cam_left_wrist": "230422271234", "cam_right_wrist": "230322274369"},
    "pc2": {"cam_high": "230422271108", "cam_left_wrist": "230322270834", "cam_right_wrist": "230422271279"},
}


def device_info(cam: RealSenseCamera) -> dict:
    dev = cam.rs_profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    d_prof = cam.rs_profile.get_stream(rs.stream.depth).as_video_stream_profile()
    c_prof = cam.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
    ext = d_prof.get_extrinsics_to(c_prof)
    di, ci = d_prof.get_intrinsics(), c_prof.get_intrinsics()
    return {
        "depth_scale_m": depth_sensor.get_depth_scale(),
        "depth_to_color_translation_m": list(ext.translation),
        "depth_to_color_rotation_is_identity": bool(np.allclose(np.array(ext.rotation).reshape(3, 3), np.eye(3), atol=1e-4)),
        "depth_intrinsics": {"fx": di.fx, "fy": di.fy, "ppx": di.ppx, "ppy": di.ppy},
        "color_intrinsics": {"fx": ci.fx, "fy": ci.fy, "ppx": ci.ppx, "ppy": ci.ppy},
    }


def grab(cam: RealSenseCamera) -> tuple[np.ndarray, np.ndarray, float]:
    cam.async_read(timeout_ms=1000)  # 새 프레임까지 기다린다
    with cam.frame_lock:
        return cam.latest_color_frame.copy(), cam.latest_depth_frame.copy(), cam.latest_timestamp


def stats(depths_raw: np.ndarray, scale: float) -> dict:
    last = depths_raw[-1].astype(np.float64) * scale
    valid = last > 0
    out = {"valid_ratio": float(valid.mean())}
    if valid.any():
        p = np.percentile(last[valid], [5, 25, 50, 75, 95])
        out["depth_m_p5_p25_p50_p75_p95"] = [round(float(x), 4) for x in p]
        out["frac_valid_in_7_50cm"] = float(((last >= 0.07) & (last <= 0.50))[valid].mean())
    # 전 프레임에서 유효한 픽셀만으로 시간 노이즈
    stack = depths_raw.astype(np.float64) * scale
    always = (stack > 0).all(axis=0)
    if always.sum() > 100:
        std = stack[:, always].std(axis=0)
        out["temporal_std_mm_median"] = round(float(np.median(std) * 1000), 3)
        out["temporal_std_mm_p90"] = round(float(np.percentile(std, 90) * 1000), 3)
    out["valid_every_frame_ratio"] = float(always.mean())
    return out


def colorize(d_m: np.ndarray, near: float, far: float, iso: float) -> np.ndarray:
    """가까울수록 빨강(TURBO 역순). 무효(0)는 검정, 범위 밖은 회색. iso m 간격 등고선."""
    t = np.clip((d_m - near) / (far - near), 0, 1)
    vis = cv2.applyColorMap(((1 - t) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[(d_m > far) | ((d_m > 0) & (d_m < near))] = (90, 90, 90)
    vis[d_m <= 0] = 0
    if iso > 0:
        # 노이즈(1.2~1.4 m 에서 σ 3~4 cm)에 흔들리지 않게 유효 픽셀만 가우시안으로 다듬은 뒤 레벨 경계를 그린다
        valid = (d_m > 0).astype(np.float32)
        num = cv2.GaussianBlur(d_m.astype(np.float32) * valid, (0, 0), 6)
        den = cv2.GaussianBlur(valid, (0, 0), 6)
        smooth = np.where(den > 0.3, num / np.maximum(den, 1e-6), 0)
        level = np.floor(smooth / iso).astype(np.int32)
        level[smooth <= 0] = -1
        edge = np.zeros(level.shape, bool)
        edge[:, 1:] |= (level[:, 1:] != level[:, :-1]) & (level[:, 1:] >= 0) & (level[:, :-1] >= 0)
        edge[1:, :] |= (level[1:, :] != level[:-1, :]) & (level[1:, :] >= 0) & (level[:-1, :] >= 0)
        vis[edge] = (255, 255, 255)
    return vis


def colorbar(height: int, near: float, far: float, iso: float, width: int = 70) -> np.ndarray:
    t = np.linspace(0, 1, height)[:, None]  # 위 = near
    bar = cv2.applyColorMap(((1 - t) * 255).astype(np.uint8).repeat(18, 1), cv2.COLORMAP_TURBO)
    out = np.full((height, width, 3), 255, np.uint8)
    out[:, :18] = bar
    step = iso if iso > 0 else (far - near) / 4
    v = np.ceil(near / step) * step
    while v <= far + 1e-6:
        y = int((v - near) / (far - near) * (height - 1))
        cv2.line(out, (14, y), (22, y), (0, 0, 0), 1)
        cv2.putText(out, f"{v:.2f}m", (24, min(max(y + 4, 10), height - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        v += step
    return out


def save_images(out: Path, name: str, color: np.ndarray, depth_raw: np.ndarray, scale: float,
                rng: tuple[float, float, float]) -> None:
    """color 는 RGB. rng = (near, far, iso) m."""
    near, far, iso = rng
    bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    vis = colorize(depth_raw.astype(np.float32) * scale, near, far, iso)
    bar = colorbar(vis.shape[0], near, far, iso)
    cv2.imwrite(str(out / f"{name}_color.png"), bgr)
    cv2.imwrite(str(out / f"{name}_depth.png"), np.hstack([vis, bar]))
    cv2.imwrite(str(out / f"{name}_overlay.png"), np.hstack([cv2.addWeighted(bgr, 0.5, vis, 0.5, 0), bar]))
    np.save(out / f"{name}_depth_raw.npy", depth_raw)


# 카메라별 컬러맵 범위 (near, far, 등고선 간격) m
#   cam_high: 베이스 접근용 — 10/27 실측 장면 0.7~2.2 m (랙 1.17 · 책상 앞 모서리 0.95 · 벽 1.87)
#   손목: 파지용 — D405 근거리 권장 범위
RANGES = {
    "cam_high": (0.5, 2.5, 0.5),
    "cam_left_wrist": (0.07, 0.6, 0.05),
    "cam_right_wrist": (0.07, 0.6, 0.05),
}


def render(run_dir: Path, scale: float) -> None:
    """이미 저장된 런을 현재 RANGES 로 다시 그린다 (카메라 무접촉)."""
    for name, rng in RANGES.items():
        raw, col = run_dir / f"{name}_depth_raw.npy", run_dir / f"{name}_color.png"
        if raw.exists() and col.exists():
            rgb = cv2.cvtColor(cv2.imread(str(col)), cv2.COLOR_BGR2RGB)
            save_images(run_dir, name, rgb, np.load(raw), scale, rng)
            print(f"다시 그림: {run_dir / name}_depth.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", default=os.environ.get("TROSSEN_PC", "pc1"), choices=list(SERIALS))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="outputs/depth_probe")
    ap.add_argument("--render", help="저장된 런 디렉터리를 다시 그리기만 한다 (stats.json 의 depth_scale 사용)")
    args = ap.parse_args()

    if args.render:
        run_dir = Path(args.render)
        st = json.loads((run_dir / "stats.json").read_text())
        render(run_dir, next(iter(st["cams"].values()))["depth_scale_m"])
        return

    out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    cams = {}
    for name, sn in SERIALS[args.pc].items():
        cfg = RealSenseCameraConfig(
            serial_number_or_name=sn, fps=args.fps, width=args.width, height=args.height, use_depth=True
        )
        cams[name] = RealSenseCamera(cfg)
    report = {"pc": args.pc, "frames": args.frames, "res": [args.width, args.height, args.fps], "cams": {}}
    try:
        for name, cam in cams.items():  # 3대 다 켠 상태에서 재야 USB 대역폭이 반영된다
            cam.connect()
        for name, cam in cams.items():
            colors, depths, ts = [], [], []
            for _ in range(args.frames):
                c, d, t = grab(cam)
                colors.append(c)
                depths.append(d)
                ts.append(t)
            depths = np.stack(depths)
            info = device_info(cam)
            dt = np.diff(ts)
            r = {
                "serial": cam.serial_number,
                "color_shape": list(colors[-1].shape),
                "depth_shape_dtype": [list(depths.shape[1:]), str(depths.dtype)],
                "measured_fps": round(float(1 / dt.mean()), 2) if len(dt) else None,
                **info,
                **stats(depths, info["depth_scale_m"]),
            }
            report["cams"][name] = r
            save_images(out, name, colors[-1], depths[-1], info["depth_scale_m"], RANGES[name])
            print(f"[{name}] {json.dumps(r, ensure_ascii=False)}")
    finally:
        for cam in cams.values():
            if cam.is_connected:
                cam.disconnect()
    (out / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
