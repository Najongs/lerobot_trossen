"""녹화 때의 실제 틱 시각에 맞춰 에피소드를 재생한다 (scripts/test_ep1.sh replay 가 부른다).

lerobot-replay 는 데이터셋 표기 fps(21)로 고정 재생한다. 그런데 리더암 녹화는 실제로 ≈19.5 Hz 로
돈다(베이스 serial re-arm 이 정책 구간에서만 켜져서) — 그대로면 재생이 7~8 % 빨라지고 베이스는
같은 속도를 더 짧게 받아 덜 움직인다(10/26 PC2 TEST_EP1: 녹화 33.6 s · 재생 31.1 s).

여기선 녹화 때 남긴 베이스 로그(LEROBOT_BASE_VEL_LOG, 한 줄 = 한 틱)의 t_mono 로 프레임 i 를
「재생 시작 + (녹화 t_i − 녹화 t_0)」 에 보낸다. 절대 시각 기준이라 틱마다 밀린 시간이 쌓이지 않는다.
로그 행 수가 에피소드 프레임 수와 다르면(여러 에피소드·리셋 구간이 섞인 로그) 거부한다.
--timing_csv 를 비우면 lerobot-replay 와 같은 고정 fps 재생이다.

그 밖은 lerobot-replay 와 같다: 같은 action processor, 틱마다 get_observation → send_action.

방향 보정 (--heading_kp > 0, --timing_csv 필요)
---------------------------------------------
재생은 속도 명령만 보내므로(open-loop) 빠른 직진의 가속 순간에 생기는 작은 요(yaw) 오차가 그대로
쌓인다(10/26 TEST_EP2: 총 회전 0.3 %·거리 1.4 % 오차인데 방향은 4° 틀어짐). 그래서 회전 명령에

    ω_cmd = ω_녹화 + clip(Kp · (θ_목표[i − L] − θ_현재), ±상한)      (|오차| < 데드밴드면 보정 0)

을 쓴다. θ_목표 = 녹화 로그의 실측 회전 속도를 녹화 시각으로 적분, θ_현재 = 재생 중 실측 회전 속도
(get_observation 이 매 틱 읽는 mobileai._latest_base_velocity)를 실제 시간으로 적분. 둘 다 베이스
바퀴 오도메트리라 바퀴가 미끄러진 만큼은 못 잡는다. L(기본 3 틱 ≈ 150 ms)은 명령→실측 지연:
목표를 그만큼 늦춰 비교해야 회전 중의 정상 지연을 오차로 보고 더 돌리지 않는다.
전진(x.vel)은 손대지 않는다.

최소 회전 속도 (--heading_min_rate_deg_s > 0)
--------------------------------------------
베이스는 정지에서 작은 회전 명령을 무시한다(10/26 TEST_EP3 재생: 3°/s 미만 명령 354 틱 합 −11.2° →
실측 −0.9°). 그래서 정지 중 남은 1~2° 오차에 Kp 만 쓰면 명령만 쌓이고 방향은 그대로다. 녹화 명령이
정지(|x.vel| < 0.03 m/s, |θ.vel| < 최소)일 때 오차가 시작 문턱(기본 1°) 을 넘으면 회전 명령을 최소
회전 속도로 올리고, 오차가 데드밴드(0.5°) 안에 들거나 부호가 바뀌면 내린다(히스테리시스 — 문턱 하나면
지연 3 틱 동안 넘어가 왕복 떨림이 난다). 값은 scripts/measure_base.py deadband 로 잰다.
"""

import csv
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


@dataclass
class TimedReplayDatasetConfig:
    repo_id: str
    episode: int = 0
    root: str | Path | None = None


@dataclass
class TimedReplayConfig:
    robot: RobotConfig
    dataset: TimedReplayDatasetConfig
    # 녹화 때의 LEROBOT_BASE_VEL_LOG CSV. 비우면 데이터셋 fps 로 고정 재생.
    timing_csv: str | None = None
    # 방향 보정 비례 이득 (1/s). 0 이면 끔(open-loop).
    heading_kp: float = 0.0
    # 명령→실측 지연(틱). 목표 방향을 이만큼 늦춰 비교한다.
    heading_lag_ticks: int = 3
    # 보정 회전 속도 상한(°/s)과 데드밴드(°).
    heading_max_correction_deg_s: float = 15.0
    heading_deadband_deg: float = 0.5
    # 정지 중 보정에 쓰는 최소 회전 속도(°/s) — 베이스 데드밴드 위. 0 이면 끔.
    heading_min_rate_deg_s: float = 0.0
    # 최소 회전 속도를 켜는 오차 문턱(°). 끄는 쪽은 heading_deadband_deg.
    heading_min_rate_start_deg: float = 1.0


def load_record_log(path: str) -> tuple[list[float], list[float]]:
    """(틱 시각 t_mono, 그 틱까지의 실측 방향 rad) — 방향은 틱 i 직전까지의 적분."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    times = [float(row["t_mono"]) for row in rows]
    headings = [0.0]
    for row, t0, t1 in zip(rows, times, times[1:]):
        headings.append(headings[-1] + float(row["meas_theta_vel"]) * (t1 - t0))
    return times, headings


class HeadingCorrector:
    """재생 중 실측 방향을 녹화 방향에 맞추는 P 제어 (모듈 docstring)."""

    # 녹화 명령이 이보다 느린 전진이면 「정지」로 본다 (최소 회전 속도 판단용)
    STILL_X_VEL = 0.03

    def __init__(self, reference: list[float], kp: float, lag_ticks: int,
                 max_correction_deg_s: float, deadband_deg: float,
                 min_rate_deg_s: float = 0.0, min_rate_start_deg: float = 1.0):
        from lerobot_robot_trossen import mobileai

        self.mobileai = mobileai
        self.reference = reference
        self.kp = kp
        self.lag_ticks = lag_ticks
        self.max_correction = math.radians(max_correction_deg_s)
        self.deadband = math.radians(deadband_deg)
        self.min_rate = math.radians(min_rate_deg_s)
        self.min_rate_start = math.radians(min_rate_start_deg)
        self.min_rate_sign = 0.0   # 최소 회전 속도로 돌리는 중이면 그 방향(±1)
        self.min_rate_ticks = 0
        self.heading = 0.0
        self.last_time: float | None = None
        self.last_rate = 0.0
        self.max_error = 0.0
        self.max_applied = 0.0
        self.saturated_ticks = 0

    def correct(self, idx: int, now: float, recorded_theta_vel: float,
                recorded_x_vel: float = 0.0) -> float:
        """get_observation 직후에 부른다. 보정된 theta.vel(rad/s)을 돌려준다."""
        if self.last_time is not None:
            self.heading += self.last_rate * (now - self.last_time)
        self.last_time = now
        self.last_rate = float(self.mobileai._latest_base_velocity["theta.vel"])

        target = self.reference[max(idx - self.lag_ticks, 0)]
        error = target - self.heading
        self.max_error = max(self.max_error, abs(error))

        still = (
            self.min_rate > 0
            and abs(recorded_x_vel) < self.STILL_X_VEL
            and abs(recorded_theta_vel) < self.min_rate
        )
        if not still or abs(error) < self.deadband or error * self.min_rate_sign < 0:
            self.min_rate_sign = 0.0
        elif self.min_rate_sign == 0.0 and abs(error) >= self.min_rate_start:
            self.min_rate_sign = math.copysign(1.0, error)

        if abs(error) < self.deadband:
            return recorded_theta_vel
        correction = self.kp * error
        if abs(correction) > self.max_correction:
            correction = math.copysign(self.max_correction, correction)
            self.saturated_ticks += 1
        command = recorded_theta_vel + correction
        if self.min_rate_sign != 0.0 and abs(command) < self.min_rate:
            command = self.min_rate_sign * self.min_rate
            correction = command - recorded_theta_vel
            self.min_rate_ticks += 1
        self.max_applied = max(self.max_applied, abs(correction))
        return command

    def summary(self) -> str:
        final_error = self.reference[-1] - self.heading
        return (
            f"방향 보정: 끝 오차 {math.degrees(final_error):+.1f}° · 최대 오차 "
            f"{math.degrees(self.max_error):.1f}° · 최대 보정 {math.degrees(self.max_applied):.1f}°/s · "
            f"상한 걸린 틱 {self.saturated_ticks} · 최소 회전 속도 틱 {self.min_rate_ticks}"
        )


@parser.wrap()
def replay(cfg: TimedReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(
        cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode]
    )
    episode_frames = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == cfg.dataset.episode
    )
    actions = episode_frames.select_columns(ACTION)
    names = dataset.features[ACTION]["names"]
    frame_count = len(episode_frames)

    if cfg.heading_kp > 0 and not cfg.timing_csv:
        raise ValueError("방향 보정(--heading_kp)은 녹화 로그(--timing_csv)가 있어야 한다")
    corrector = None
    if cfg.timing_csv:
        tick_times, reference_headings = load_record_log(cfg.timing_csv)
        if len(tick_times) != frame_count:
            raise ValueError(
                f"{cfg.timing_csv} 는 {len(tick_times)} 틱인데 에피소드 {cfg.dataset.episode} 는 "
                f"{frame_count} 프레임 — 이 에피소드의 녹화 로그가 아니다 (고정 fps 재생은 TIMING=fixed)"
            )
        offsets = [t - tick_times[0] for t in tick_times]
        logging.info(
            f"녹화 시각 재생: {frame_count} 프레임 · {offsets[-1]:.1f} s · "
            f"평균 {(frame_count - 1) / offsets[-1]:.2f} Hz (표기 {dataset.fps} fps)"
        )
        if cfg.heading_kp > 0:
            corrector = HeadingCorrector(
                reference_headings, cfg.heading_kp, cfg.heading_lag_ticks,
                cfg.heading_max_correction_deg_s, cfg.heading_deadband_deg,
                cfg.heading_min_rate_deg_s, cfg.heading_min_rate_start_deg,
            )
            logging.info(
                f"방향 보정 켬: Kp {cfg.heading_kp} · 지연 {cfg.heading_lag_ticks} 틱 · 상한 "
                f"{cfg.heading_max_correction_deg_s}°/s · 데드밴드 {cfg.heading_deadband_deg}° · "
                f"정지 중 최소 {cfg.heading_min_rate_deg_s}°/s(오차 ≥ {cfg.heading_min_rate_start_deg}°) · "
                f"녹화 끝 방향 {math.degrees(reference_headings[-1]):+.1f}°"
            )
    else:
        offsets = [i / dataset.fps for i in range(frame_count)]
        logging.info(f"고정 fps 재생: {frame_count} 프레임 · {dataset.fps} fps")

    robot.connect()
    try:
        start = time.perf_counter()
        late_ticks = 0
        for idx in range(frame_count):
            wait = start + offsets[idx] - time.perf_counter()
            if wait > 0:
                precise_sleep(wait)
            elif wait < -0.05:
                late_ticks += 1

            action_array = actions[idx][ACTION]
            action = {name: action_array[i] for i, name in enumerate(names)}
            robot_obs = robot.get_observation()
            if corrector is not None:
                action["theta.vel"] = corrector.correct(
                    idx, time.perf_counter(), float(action["theta.vel"]),
                    float(action["x.vel"]),
                )
            processed_action = robot_action_processor((action, robot_obs))
            robot.send_action(processed_action)
        elapsed = time.perf_counter() - start
        logging.info(
            f"재생 끝: {elapsed:.1f} s (목표 {offsets[-1]:.1f} s) · 50 ms 넘게 늦은 틱 {late_ticks}"
        )
        if corrector is not None:
            logging.info(corrector.summary())
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
