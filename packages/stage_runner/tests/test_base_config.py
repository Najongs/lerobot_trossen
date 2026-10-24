"""The schema and vocabulary the base_replay / base_align stages add.

Covers what the YAML can say (StageConfig.replay / .align and their defaults),
that both executors are in the dispatch table, the two new terminators and
event names, the StageContext field, and the stage ceiling floors that preflight
and base_assets check a YAML against. The executors themselves are tested where
they land.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from stage_runner import base_stages, config, executors, record_adapter
from stage_runner.config import (
    BASE_EXECUTORS,
    EXECUTOR_BASE_ALIGN,
    EXECUTOR_BASE_REPLAY,
    POLICY_EXECUTORS,
    BaseAlignConfig,
    BaseReplayConfig,
    StageConfig,
)
from stage_runner.context import StageContext
from stage_runner.events import (
    EVENT_NAMES,
    EVENT_STAGE_BASE_ALIGN,
    EVENT_STAGE_BASE_ALIGN_STEP,
    EVENT_STAGE_BASE_REPLAY,
    EventLog,
)
from stage_runner.results import (
    ABORTING_TERMINATORS,
    TERMINATED_BY_BASE_FAILED,
    TERMINATED_BY_CONVERGED,
    TERMINATED_BY_VALUES,
)

CONFIGS = Path(__file__).parent / "configs"

YAML = """\
version: 1
robot:
  type: stage_runner_mock_robot
  id: unit
  arm_layout: bimanual
  simulate_base: true
  scene_camera: cam_high
dataset:
  repo_id: local/eval_base_config
  video: false
stages:
  - id: s1
    name: drive to the rack
    executor: base_replay
    terminator: {type: timeout, timeout_s: 30}
    replay:
      dataset: kiroaiseoul/test_replay_timing_1024_basepose
      episode: 0
      until_row: 297
  - id: s1_align
    name: align to the rack
    executor: base_align
    terminator: {type: timeout, timeout_s: 60}
    align:
      reference_dataset: kiroaiseoul/test_replay_timing_1024
      episode: 0
      frame: 302
      landmark_roi: [285, 197, 93, 61]
      landmark_distance_m: 0.60
      camera_matrix: [[392.6346, 0, 322.7227], [0, 392.0662, 236.2555], [0, 0, 1]]
      dist_coeffs: [0, 0, 0, 0, 0]
  - id: s2
    name: mock_stage_two
    policy_path: mock://hold
    terminator: {type: timeout, timeout_s: 1.0}
"""


def _parse(tmp_path: Path, text: str = YAML) -> config.StageRunnerConfig:
    record_adapter.register_plugins()
    path = tmp_path / "chain.yaml"
    path.write_text(text, encoding="utf-8")
    return config.parse_config(["--config_path", str(path)])


def test_the_blocks_parse_with_field_defaults(tmp_path: Path) -> None:
    parsed = _parse(tmp_path)
    replay, align, policy = parsed.stages
    assert replay.replay == BaseReplayConfig(
        dataset="kiroaiseoul/test_replay_timing_1024_basepose",
        episode=0,
        until_row=297,
    )
    assert replay.align is None
    # The field's replay rate, the field's tolerances and move budget, and the
    # coarse heading fallback ON, as on 10/24.
    assert replay.replay.row_period_s == pytest.approx(1.0 / 21.0)
    assert replay.replay.from_row == 0
    assert align.replay is None
    assert align.align is not None
    assert align.align.yaw_tol_deg == 2.0
    assert align.align.dist_tol_cm == 1.0
    assert align.align.max_moves == 6
    assert align.align.coarse_heading is True
    assert align.align.distance_width_fallback is False
    assert align.align.min_turn_rate_rad_s == 0.0
    assert align.align.turn_kp == 2.0
    assert align.align.camera == "cam_high"
    assert align.align.landmark_roi == [285, 197, 93, 61]
    assert all(isinstance(value, int) for value in align.align.landmark_roi)
    assert align.align.camera_matrix[0] == [392.6346, 0.0, 322.7227]
    assert align.align.camera_up == pytest.approx(
        [-0.06288349, -0.84574326, -0.52987169]
    )
    assert policy.replay is None and policy.align is None


def test_coarse_heading_can_be_switched_off(tmp_path: Path) -> None:
    text = YAML.replace(
        "      dist_coeffs: [0, 0, 0, 0, 0]\n",
        "      dist_coeffs: [0, 0, 0, 0, 0]\n      coarse_heading: false\n",
    )
    assert _parse(tmp_path, text).stages[1].align.coarse_heading is False


def test_list_defaults_are_not_shared_between_stages() -> None:
    first, second = BaseAlignConfig(), BaseAlignConfig()
    first.camera_up[0] = 9.0
    first.landmark_roi.append(1)
    assert second.camera_up[0] == pytest.approx(-0.06288349)
    assert second.landmark_roi == []


def test_an_old_yaml_parses_with_no_base_blocks() -> None:
    record_adapter.register_plugins()
    smoke = config.parse_config(["--config_path", str(CONFIGS / "smoke_mock.yaml")])
    assert all(stage.replay is None and stage.align is None for stage in smoke.stages)
    # Not popped from the snapshot (see the version note in config.py): every
    # stage shows the two nulls, which is cosmetic.
    text = config.resolved_config_text(smoke)
    assert text.count("replay: null") == len(smoke.stages)
    assert text.count("align: null") == len(smoke.stages)


def test_both_base_executors_are_dispatchable_and_run_no_policy() -> None:
    assert BASE_EXECUTORS == {EXECUTOR_BASE_REPLAY, EXECUTOR_BASE_ALIGN}
    assert executors.EXECUTORS[EXECUTOR_BASE_REPLAY] is executors.run_base_replay_stage
    assert executors.EXECUTORS[EXECUTOR_BASE_ALIGN] is executors.run_base_align_stage
    assert not BASE_EXECUTORS & POLICY_EXECUTORS


class _Context:
    """The two attributes the wrappers read before they call base_stages."""

    def __init__(self, **events: bool) -> None:
        self.events = {"exit_early": False, **events}


@pytest.mark.parametrize(
    ("wrapper", "executor"),
    [
        (executors.run_base_replay_stage, EXECUTOR_BASE_REPLAY),
        (executors.run_base_align_stage, EXECUTOR_BASE_ALIGN),
    ],
)
def test_the_wrappers_honour_the_flags_then_call_the_stage(
    wrapper, executor, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = StageConfig(id="b", name="base", executor=executor)
    aborted = wrapper(_Context(stop_recording=True), stage)
    assert aborted.terminated_by == "stop_recording"
    assert aborted.frames == 0

    seen: list[bool] = []
    target = "replay" if executor == EXECUTOR_BASE_REPLAY else "align"
    monkeypatch.setattr(
        base_stages,
        target,
        lambda context, stage: seen.append(context.events["exit_early"]),
    )
    wrapper(_Context(exit_early=True), stage)
    # A right arrow from the inter-stage gap is discarded before the stage runs.
    assert seen == [False]


def test_the_new_terminators_and_events_are_in_the_closed_vocabularies(
    tmp_path: Path,
) -> None:
    assert TERMINATED_BY_CONVERGED == "converged"
    assert TERMINATED_BY_BASE_FAILED == "base_failed"
    assert {TERMINATED_BY_CONVERGED, TERMINATED_BY_BASE_FAILED} <= set(
        TERMINATED_BY_VALUES
    )
    # An align failure must not let the next policy run; converging must.
    assert TERMINATED_BY_BASE_FAILED in ABORTING_TERMINATORS
    assert TERMINATED_BY_CONVERGED not in ABORTING_TERMINATORS
    assert len(set(TERMINATED_BY_VALUES)) == len(TERMINATED_BY_VALUES)

    names = (
        EVENT_STAGE_BASE_REPLAY,
        EVENT_STAGE_BASE_ALIGN,
        EVENT_STAGE_BASE_ALIGN_STEP,
    )
    assert names == ("stage_base_replay", "stage_base_align", "stage_base_align_step")
    assert set(names) <= set(EVENT_NAMES)
    with EventLog(tmp_path / "events.jsonl", "run") as log:
        for name in names:
            log.emit(name, stage_id="b", terminator=TERMINATED_BY_BASE_FAILED)


def test_the_context_carries_no_base_assets_by_default() -> None:
    field = StageContext.__dataclass_fields__["base_assets"]
    assert field.default_factory() == {}


def test_the_ceiling_floors() -> None:
    # The align floor at the default 6 moves is 56.5 s -- "57 s" in the refusal.
    assert base_stages.align_ceiling_floor_s(6) == pytest.approx(56.5)
    assert math.ceil(base_stages.align_ceiling_floor_s(6)) == 57
    # stage1_align's stage 1 of episode 0: 298 rows at 21 Hz.
    assert base_stages.replay_ceiling_floor_s(298, 1.0 / 21.0) == pytest.approx(
        0.5 + 297 / 21.0 + 4.0 + 2.0
    )


def test_the_longest_move_ramp_covers_every_move_the_estimates_allow() -> None:
    landmark_align = pytest.importorskip("stage_runner.landmark_align")
    turns = [landmark_align.MAX_ABS_YAW_DEG, landmark_align.MAX_ABS_COARSE_YAW_DEG]
    longest = max(
        math.radians(max(turns)) / base_stages.TURN_RATE_RAD_S,
        landmark_align.MAX_ABS_MOVE_M / base_stages.DRIVE_RATE_M_S,
    )
    assert base_stages.MOVE_RAMP_MAX_S >= longest
    assert base_stages.MOVE_RAMP_MAX_S == pytest.approx(longest)


def test_importing_the_executors_does_not_import_the_estimator() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import stage_runner.executors; "
        "print('stage_runner.landmark_align' in sys.modules, "
        "'stage_runner.mock_scene' in sys.modules)"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert output == ["False", "False"]
