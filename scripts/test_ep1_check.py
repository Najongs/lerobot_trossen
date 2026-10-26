"""재생이 녹화 끝과 같은 자리·방향에 섰는지 cam_high 영상으로 본다 (scripts/test_ep1.sh check 가 부른다).

  uv run python scripts/test_ep1_check.py --repo kiroaiseoul/TEST_EP4 --serial 230422271108

녹화 에피소드의 cam_high 마지막 프레임과 지금 cam_high 한 장을 ORB 특징점으로 맞추고
(RANSAC, 회전·균일 확대·평행이동만 허용), 그 변환을 읽는다:

- 좌우 이동(px) → 방향 오차 ≈ atan(dx / fx). 카메라 내부 파라미터 fx 는 RealSense 에서 읽는다
  (PC2 cam_high 640×480: fx ≈ 392.7 px, 가로 화각 ≈ 78°). 장면이 오른쪽으로 밀렸으면 로봇이 왼쪽으로
  더 돌아 있는 것이다. 옆으로 비켜 선 것(횡이동)도 같은 꼴로 보여 섞인다 — 가까운 물체일수록 커진다.
- 확대(배) → 앞뒤. 1 보다 크면 장면이 커 보인다 = 더 앞에 섰다. 단일 카메라라 cm 는 알 수 없다.

cam_high 엔 로봇 자기 몸체·팔이 찍힌다. 로봇이 어디 서든 화면에서 그대로라 「0 px 이동」 대응점이 되어
오차를 0 으로 끌어당긴다. 그래서 녹화 에피소드에서 프레임을 고르게 뽑아, 베이스가 움직이는 동안에도
거의 안 변한 픽셀(로봇 몸체)을 가리고 특징점을 찾는다. 베이스가 거의 안 움직인 에피소드면 가릴 곳을
못 정한다(경고).

바퀴 오도메트리와 무관한 바닥 기준이라 미끄러짐까지 들어간다. 녹화·지금 두 장과 겹친 그림을 남긴다.
녹화 끝 이후 장면이 바뀌었으면(사람·물건) 대응점이 줄고 값이 흔들린다 — 대응점 수를 같이 본다.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

# 대응점(RANSAC 안쪽)이 이보다 적으면 값을 믿지 않는다
MIN_INLIERS = 30
# 로봇 몸체 가림: 에피소드에서 뽑을 프레임 수, 「안 변함」 표준편차 문턱(0..255), 가림을 넓히는 폭(px)
MASK_FRAMES = 12
MASK_STD = 10.0
MASK_DILATE = 15
# 가림이 화면의 이만큼을 넘으면 베이스가 거의 안 움직인 에피소드로 보고 가리지 않는다
MASK_MAX_FRACTION = 0.6


def _to_rgb(image) -> np.ndarray:
    return (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def recorded_frames(repo: str, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """(에피소드 cam_high 마지막 프레임 RGB, 로봇 몸체 가림 — 특징점을 찾을 곳이 255).

    가림 = 고르게 뽑은 MASK_FRAMES 장에서 픽셀 밝기 표준편차가 MASK_STD 미만인 곳(넓혀서) 을 뺀 나머지.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo, episodes=[episode])
    count = len(dataset)
    last = _to_rgb(dataset[count - 1]["observation.images.cam_high"])
    picks = np.linspace(0, count - 1, MASK_FRAMES).round().astype(int)
    stack = np.stack([
        cv2.GaussianBlur(cv2.cvtColor(_to_rgb(dataset[int(i)]["observation.images.cam_high"]),
                                      cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
        for i in picks
    ])
    robot = (stack.std(axis=0) < MASK_STD).astype(np.uint8)
    robot = cv2.dilate(robot, np.ones((MASK_DILATE, MASK_DILATE), np.uint8))
    fraction = float(robot.mean())
    if fraction > MASK_MAX_FRACTION:
        print(f"!! 에피소드 동안 화면이 {fraction * 100:.0f} % 거의 안 변했다 — 베이스가 거의 안 움직여 "
              "로봇 몸체를 가려 낼 수 없다. 가리지 않는다(값이 0 쪽으로 끌릴 수 있다)")
        return last, np.full(robot.shape, 255, np.uint8)
    return last, np.where(robot > 0, 0, 255).astype(np.uint8)


def capture_now(serial: str, warmup: int = 15) -> tuple[np.ndarray, float]:
    """지금 cam_high 한 장 (RGB uint8)과 fx. 처음 몇 장은 노출이 안정되기 전이라 버린다."""
    import pyrealsense2 as rs

    # 한동안 안 쓴 D405 는 첫 start 에서 프레임이 안 오고 다시 열면 바로 온다(10/26 PC2) — 다시 연다
    for attempt in range(3):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = pipeline.start(config)
        try:
            fx = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics().fx
            for _ in range(warmup):
                frames = pipeline.wait_for_frames(5000)
            return np.asanyarray(frames.get_color_frame().get_data()).copy(), fx
        except RuntimeError as error:
            print(f"   cam_high 프레임 안 옴 ({error}) — 다시 연다 ({attempt + 1}/3)")
        finally:
            pipeline.stop()
    raise SystemExit(f"cam_high({serial}) 에서 프레임을 못 받았다 — USB 연결 확인")


def estimate(recorded: np.ndarray, now: np.ndarray, mask: np.ndarray | None = None):
    """recorded → now 의 부분 아핀(회전·균일 확대·평행이동). (행렬, 안쪽 대응점 수, 전체 대응 수).
    mask(255 = 찾을 곳)는 두 장 모두에 건다 — 로봇 몸체는 두 장에서 같은 자리다."""
    orb = cv2.ORB_create(nfeatures=3000)
    gray_a = cv2.cvtColor(recorded, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(now, cv2.COLOR_RGB2GRAY)
    keys_a, desc_a = orb.detectAndCompute(gray_a, mask)
    keys_b, desc_b = orb.detectAndCompute(gray_b, mask)
    if desc_a is None or desc_b is None:
        return None, 0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(desc_a, desc_b, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 6:
        return None, 0, len(good)
    points_a = np.float32([keys_a[m.queryIdx].pt for m in good])
    points_b = np.float32([keys_b[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        points_a, points_b, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return matrix, int(inliers.sum()) if inliers is not None else 0, len(good)


def describe(matrix: np.ndarray, fx: float, width: int, height: int) -> dict:
    """변환을 방향 오차(°)·확대·롤(°)로. 좌우 이동은 화면 중심점이 옮겨 간 양으로 잰다."""
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    roll = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    center = np.array([width / 2, height / 2, 1.0])
    moved = matrix @ center
    dx, dy = moved[0] - center[0], moved[1] - center[1]
    # 장면이 오른쪽(+dx)으로 갔다 = 카메라(로봇)가 왼쪽으로 더 돌아 있다
    yaw_left = math.degrees(math.atan2(dx, fx))
    return {"yaw_left_deg": yaw_left, "scale": scale, "roll_deg": roll, "dx_px": dx, "dy_px": dy}


def save_figure(path: Path, recorded: np.ndarray, now: np.ndarray, text: str,
                mask: np.ndarray | None = None) -> None:
    """[녹화 끝 | 지금 | 겹침] 한 장. 겹침은 녹화=빨강·지금=청록 — 맞으면 회색으로 겹친다."""
    gray_a = cv2.cvtColor(recorded, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(now, cv2.COLOR_RGB2GRAY)
    overlay = np.dstack([gray_b, gray_b, gray_a])  # BGR: 지금 → B·G, 녹화 → R
    panels = [cv2.cvtColor(recorded, cv2.COLOR_RGB2BGR), cv2.cvtColor(now, cv2.COLOR_RGB2BGR), overlay]
    if mask is not None:
        # 녹화 끝 그림에서 가린 곳(로봇 몸체)을 어둡게
        panels[0][mask == 0] = (panels[0][mask == 0] * 0.35).astype(np.uint8)
    for panel, label in zip(panels, ("recorded end (dark = masked robot)", "now", "overlay (red=recorded, cyan=now)")):
        cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    figure = np.hstack(panels)
    cv2.putText(figure, text, (10, figure.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--serial", required=True, help="cam_high 시리얼 (scripts/robot_hw.sh CAM_HIGH)")
    parser.add_argument("--out", default=None, help="그림 경로 (기본 ~/eval_logs/test_ep1/<repo>_check_<시각>.png)")
    parser.add_argument("--now-image", default=None, help="카메라 대신 이 그림을 지금 장면으로 (시험용)")
    parser.add_argument("--fx", type=float, default=None, help="--now-image 일 때 fx (기본 392.7 = PC2 cam_high)")
    args = parser.parse_args()

    recorded, mask = recorded_frames(args.repo, args.episode)
    if args.now_image:
        now = cv2.cvtColor(cv2.imread(args.now_image), cv2.COLOR_BGR2RGB)
        fx = args.fx or 392.7
    else:
        now, fx = capture_now(args.serial)

    matrix, inliers, matches = estimate(recorded, now, mask)
    out = Path(args.out) if args.out else (
        Path.home() / "eval_logs" / "test_ep1"
        / f"{args.repo.replace('/', '_')}_check_{time.strftime('%m%d_%H%M%S')}.png"
    )
    if matrix is None or inliers < MIN_INLIERS:
        save_figure(out, recorded, now, f"match failed: inliers {inliers}/{matches}", mask)
        print(f"!! 대응점이 부족하다 (안쪽 {inliers} / 대응 {matches}, 기준 {MIN_INLIERS}) — "
              "장면이 바뀌었거나 너무 멀리 섰다. 그림으로 확인:")
        print(f"   {out}")
        raise SystemExit(3)

    result = describe(matrix, fx, recorded.shape[1], recorded.shape[0])
    yaw = result["yaw_left_deg"]
    side = "왼쪽" if yaw > 0 else "오른쪽"
    scale = result["scale"]
    depth = "더 앞(장면이 커 보임)" if scale > 1 else "더 뒤(장면이 작아 보임)"
    text = f"yaw {yaw:+.1f} deg (left+)  scale {scale:.3f}  inliers {inliers}"
    save_figure(out, recorded, now, text, mask)

    print(f"cam_high 비교 — {args.repo} 에피소드 {args.episode} 녹화 끝 vs 지금 (fx {fx:.1f} px)")
    print(f"  방향: 녹화 끝보다 {side}으로 {abs(yaw):.1f}° 더 돌아 있다  (화면 중심 {result['dx_px']:+.0f} px)")
    print(f"  앞뒤: 확대 {scale:.3f} 배 → {depth} {abs(scale - 1) * 100:.1f} %")
    print(f"  기울기(롤) {result['roll_deg']:+.1f}° · 세로 {result['dy_px']:+.0f} px  (평지면 ≈0)")
    print(f"  대응점 {inliers} / {matches}  (적으면 값이 흔들린다) · 로봇 몸체로 가린 화면 {(mask == 0).mean() * 100:.0f} %")
    print(f"  그림: {out}")
    print("  ※ 옆으로 비켜 선 것도 방향 오차처럼 보인다. 확대 % 는 앞뒤 방향만 — cm 는 아니다.")


if __name__ == "__main__":
    main()
