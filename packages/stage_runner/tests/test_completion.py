"""The completion classifier terminator: decision rule, the preprocessor tap, preflight, and the chain.

The decision rule and the tap are tested on their own; the chain tests drive
cli.main against the mock robot with the REAL record_loop and a stand-in model
(the mock robot has no cameras, so the stand-in reads observation.state). What
they prove is the plumbing -- frames reach the watcher only through the policy
preprocessor, a decision ends the stage before its timer with
terminated_by="classifier", shadow mode leaves the timer in charge, and the
stage_completion line is in the log. Whether a trained model decides RIGHT is
the offline evaluation's question, not this file's.

    uv run --with pytest pytest packages/stage_runner/tests/test_completion.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
from stage_runner import cli, completion, config, preflight
from stage_runner.completion import (
    CompletionMeta,
    CompletionWatcher,
    GripTracker,
    TappedPreprocessor,
    WatcherSettings,
)
from stage_runner.preflight import PreflightError

SMOKE_CONFIG = Path(__file__).parent / "configs" / "smoke_mock.yaml"
REPO_ID = "local/eval_smoke_chain"


class _FakeModel:
    """Duck-typed CompletionModel: scores whatever `scores` yields, on the CPU."""

    def __init__(self, scores: list[float] | None = None, keys=("observation.state",)):
        self.path = "fake://completion"
        self.meta = CompletionMeta(
            camera_keys=tuple(keys), threshold=0.9, hold_s=0.2, min_s=0.3
        )
        self.device = torch.device("cpu")
        self.scores = list(scores or [])
        self.prepared = 0

    def prepare(self, observation: dict[str, Any]) -> list[torch.Tensor]:
        self.prepared += 1
        return [observation[key] for key in self.meta.camera_keys]

    def score(self, prepared: list[torch.Tensor]) -> float:
        return self.scores.pop(0) if self.scores else 1.0


def _watcher(
    mode: str = "fire", events: dict[str, bool] | None = None
) -> CompletionWatcher:
    settings = WatcherSettings(threshold=0.9, hold_s=0.5, min_s=1.0, mode=mode)
    return CompletionWatcher(_FakeModel(), settings, {} if events is None else events)


# --- the decision rule --------------------------------------------------------


def test_fires_only_after_the_score_holds_for_hold_s() -> None:
    events: dict[str, bool] = {}
    watcher = _watcher(events=events)
    for t, score in [(1.0, 0.95), (1.2, 0.95), (1.4, 0.95)]:
        watcher.decide(t, score)
    assert not watcher.fired and "exit_early" not in events
    watcher.decide(1.5, 0.95)
    assert watcher.fired and events["exit_early"] is True
    assert watcher.decided_at_s == 1.5


def test_a_dip_below_the_threshold_restarts_the_hold() -> None:
    watcher = _watcher()
    for t, score in [(1.0, 0.95), (1.4, 0.5), (1.6, 0.95), (2.0, 0.95)]:
        watcher.decide(t, score)
    assert not watcher.fired
    watcher.decide(2.1, 0.95)
    assert watcher.fired and watcher.decided_at_s == 2.1


def test_nothing_fires_before_min_s() -> None:
    watcher = _watcher()
    for t in (0.0, 0.2, 0.4, 0.6, 0.8):
        watcher.decide(t, 0.99)
    assert not watcher.fired
    # Held since 0.0, so the first frame at or past min_s decides.
    watcher.decide(1.0, 0.99)
    assert watcher.fired and watcher.decided_at_s == 1.0


def test_shadow_mode_records_the_decision_and_leaves_the_flag_alone() -> None:
    events: dict[str, bool] = {}
    watcher = _watcher(mode="shadow", events=events)
    for t in (1.0, 1.3, 1.6):
        watcher.decide(t, 0.99)
    assert watcher.decided_at_s == 1.6
    assert not watcher.fired and "exit_early" not in events


def test_an_exit_early_already_set_is_not_claimed() -> None:
    # The space bar (intervention patch) or the right arrow got there first.
    events = {"exit_early": True}
    watcher = _watcher(events=events)
    for t in (1.0, 1.3, 1.6):
        watcher.decide(t, 0.99)
    assert not watcher.fired
    assert watcher.skipped_reason == "exit_early was already set"


def test_one_decision_per_stage() -> None:
    events: dict[str, bool] = {}
    watcher = _watcher(events=events)
    for t in (1.0, 1.3, 1.6):
        watcher.decide(t, 0.99)
    events["exit_early"] = False  # record_loop clears it on its way out
    watcher.decide(2.5, 0.99)
    assert events["exit_early"] is False
    assert watcher.decided_at_s == 1.6
    assert len(watcher.samples) == 4


# --- the tap and the thread ---------------------------------------------------


class _Pipeline:
    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def __call__(self, observation: Any) -> Any:
        self.calls += 1
        return observation

    def reset(self) -> None:
        self.resets += 1


def test_the_tap_forwards_the_call_and_reset() -> None:
    inner = _Pipeline()
    watcher = _watcher()
    tapped = TappedPreprocessor(inner, watcher)
    tapped.reset()
    observation = {"observation.state": torch.zeros(1, 3)}
    assert tapped(observation) is observation
    assert inner.resets == 1 and inner.calls == 1
    # Not started: offer is a no-op rather than a frame for a stage that has
    # not begun.
    assert watcher.offered == 0


def test_the_watcher_thread_scores_offered_frames_and_fires() -> None:
    events: dict[str, bool] = {}
    clock = {"t": 0.0}
    settings = WatcherSettings(threshold=0.9, hold_s=0.4, min_s=0.0, mode="fire")
    watcher = CompletionWatcher(
        _FakeModel(), settings, events, clock=lambda: clock["t"]
    )
    watcher.start()
    try:
        for step in range(10):
            clock["t"] = step * 0.2
            watcher.offer({"observation.state": torch.zeros(1, 3)})
            # Let the thread take each frame before the next one replaces it.
            for _ in range(200):
                if len(watcher.samples) > step:
                    break
                time.sleep(0.005)
            if watcher.fired:
                break
    finally:
        watcher.stop()
    assert watcher.fired and events["exit_early"] is True
    assert watcher.decided_at_s == pytest.approx(0.4)


def test_offer_is_throttled_to_period_s() -> None:
    clock = {"t": 0.0}
    settings = WatcherSettings(threshold=0.9, hold_s=0.4, min_s=0.0, mode="shadow")
    watcher = CompletionWatcher(_FakeModel(), settings, {}, clock=lambda: clock["t"])
    watcher.start()
    try:
        for tick in range(21):  # one second at ~21 Hz
            clock["t"] = tick / 21
            watcher.offer({"observation.state": torch.zeros(1, 3)})
    finally:
        watcher.stop()
    assert watcher.offered == 5


def test_a_missing_camera_stops_scoring_without_raising() -> None:
    model = _FakeModel(keys=("observation.images.cam_high",))
    settings = WatcherSettings(threshold=0.9, hold_s=0.0, min_s=0.0, mode="fire")
    watcher = CompletionWatcher(model, settings, {})
    watcher.start()
    watcher.offer({"observation.state": torch.zeros(1, 3)})
    watcher.stop()
    assert "could not prepare the frame" in watcher.error
    assert not watcher.fired


def test_prepare_image_is_the_size_the_head_needs() -> None:
    image = torch.rand(1, 3, 480, 640)
    out = completion.prepare_image(image)
    assert out.shape == (1, 3, 128, 128)
    # ImageNet normalization: a mid-grey frame lands near zero on every channel.
    grey = completion.prepare_image(torch.full((1, 3, 480, 640), 0.45))
    assert grey.abs().max() < 0.5


def test_meta_rejects_an_unknown_format_and_no_cameras() -> None:
    with pytest.raises(ValueError, match="format"):
        CompletionMeta.from_json(json.dumps({"format": 99}), "x")
    with pytest.raises(ValueError, match="camera_keys"):
        CompletionMeta.from_json(
            json.dumps(
                {
                    "format": 1,
                    "camera_keys": [],
                    "threshold": 0.9,
                    "hold_s": 1,
                    "min_s": 1,
                }
            ),
            "x",
        )


def test_yaml_values_override_the_checkpoint_defaults() -> None:
    meta = CompletionMeta(camera_keys=("a",), threshold=0.9, hold_s=1.0, min_s=3.0)
    terminator = config.TerminatorConfig(
        type="classifier", timeout_s=20.0, model_path="m", threshold=0.8, mode="shadow"
    )
    settings = completion.resolve_settings(terminator, meta)
    assert settings == WatcherSettings(
        threshold=0.8, hold_s=1.0, min_s=3.0, mode="shadow"
    )


# --- the grippers -------------------------------------------------------------


def _hold_grip(**overrides: Any) -> config.GripConfig:
    values = {
        "state_names": ["left", "right"],
        "hold_min_m": 0.004,
        "hold_max_m": 0.0101,
        "settle_s": 0.5,
    } | overrides
    return config.GripConfig(**values)


def _tick(tracker: GripTracker, start: float, stop: float, *widths: float) -> float:
    """Feed constant widths at 20 Hz over [start, stop); returns stop."""
    t = start
    while t < stop - 1e-9:
        tracker.update(t, list(widths))
        t += 0.05
    return stop


def test_the_hold_gate_needs_every_gripper_in_the_band_for_settle_s() -> None:
    tracker = GripTracker(_hold_grip(), [0, 1])
    _tick(tracker, 0.0, 1.0, 0.0065, 0.0)  # left holds, right closed on nothing
    assert not tracker.holding(1.0)
    _tick(tracker, 1.0, 1.4, 0.0065, 0.0066)
    assert not tracker.holding(1.35)  # right only 0.35 s in the band
    _tick(tracker, 1.4, 1.6, 0.0065, 0.0066)
    assert tracker.holding(1.55)
    _tick(tracker, 1.6, 1.65, 0.015, 0.0066)  # left opens again
    assert not tracker.holding(1.6)


def test_a_grasp_is_an_open_then_a_settled_close() -> None:
    grip = config.GripConfig(state_names=["right"], settle_s=0.5)
    tracker = GripTracker(grip, [0])
    _tick(tracker, 0.0, 1.0, 0.0)  # closed from the start: not a grasp
    assert tracker.latest_grasp_s() is None
    _tick(tracker, 1.0, 2.0, 0.013)
    _tick(tracker, 2.0, 2.3, 0.0)
    assert tracker.latest_grasp_s() is None  # not settled yet
    _tick(tracker, 2.3, 2.6, 0.0)
    assert tracker.latest_grasp_s() == pytest.approx(2.0)
    assert tracker.events == [[2.0, "right", "grasped"]]


def test_reopening_cancels_the_grasp_and_a_regrasp_restarts_it() -> None:
    grip = config.GripConfig(state_names=["right"], settle_s=0.5)
    tracker = GripTracker(grip, [0])
    _tick(tracker, 0.0, 1.0, 0.013)
    _tick(tracker, 1.0, 2.0, 0.0)
    assert tracker.latest_grasp_s() == pytest.approx(1.0)
    _tick(tracker, 2.0, 2.5, 0.013)
    assert tracker.latest_grasp_s() is None
    _tick(tracker, 2.5, 3.5, 0.0)
    assert tracker.latest_grasp_s() == pytest.approx(2.5)
    assert [e[2] for e in tracker.events] == ["grasped", "reopened", "grasped"]


def test_a_tube_width_is_neither_open_nor_a_grasp() -> None:
    # Stage 6 on t06: the left gripper sat at 5-7 mm the whole stage.
    grip = config.GripConfig(state_names=["left"], settle_s=0.5)
    tracker = GripTracker(grip, [0])
    _tick(tracker, 0.0, 5.0, 0.0071)
    assert tracker.latest_grasp_s() is None and tracker.events == []


def test_the_tracker_resolves_names_against_the_dataset_state() -> None:
    tracker = GripTracker.for_stage(_hold_grip(), ["x", "right", "left"])
    assert tracker.indices == [2, 1]
    with pytest.raises(ValueError, match="not in observation.state"):
        GripTracker.for_stage(_hold_grip(), ["left"])


def test_the_gate_holds_a_classifier_decision_until_the_grippers_settle() -> None:
    events: dict[str, bool] = {}
    watcher = _watcher(events=events)
    for t in (1.0, 1.2, 1.4, 1.6):
        watcher.decide(t, 0.95, holding=False)
    assert not watcher.fired and watcher.gate_blocked_at_s == pytest.approx(1.6)
    watcher.decide(1.8, 0.95, holding=True)  # no second hold_s wait
    assert watcher.fired and watcher.decided_at_s == pytest.approx(1.8)


def _grasp_watcher(
    mode: str = "fire", events: dict[str, bool] | None = None, wait_s: float = 2.0
) -> CompletionWatcher:
    settings = WatcherSettings(
        threshold=None, hold_s=None, min_s=None, mode=mode, rule="grasp", wait_s=wait_s
    )
    grip = GripTracker(config.GripConfig(state_names=["right"], settle_s=0.5), [0])
    return CompletionWatcher(
        None, settings, {} if events is None else events, grip=grip
    )


def _offer_widths(watcher: CompletionWatcher, clock: list[float], *steps) -> None:
    for start, stop, width in steps:
        clock[0] = start
        while clock[0] < stop - 1e-9:
            watcher.offer({"observation.state": torch.tensor([[width]])})
            clock[0] += 0.05


def test_a_grasp_stage_fires_wait_s_after_the_grasp() -> None:
    events: dict[str, bool] = {}
    watcher = _grasp_watcher(events=events)
    clock = [0.0]
    watcher.clock = lambda: clock[0]
    watcher.start()
    _offer_widths(watcher, clock, (0.0, 1.0, 0.013), (1.0, 2.9, 0.0))
    assert not watcher.fired
    _offer_widths(watcher, clock, (2.9, 3.2, 0.0))
    assert watcher.fired and events["exit_early"] is True
    assert watcher.grasp_at_s == pytest.approx(1.0)
    # the first 20 Hz tick at or past grasp + wait_s
    assert watcher.grasp_decided_at_s == pytest.approx(3.0, abs=0.06)
    assert "grasp at 1.00s + wait 2.00s" in watcher.describe_fire()
    summary = watcher.summary()
    assert summary["rule"] == "grasp" and summary["grip_events"][0][2] == "grasped"
    assert summary["grip_samples"] and summary["frames_scored"] == 0


def test_a_grasp_stage_in_shadow_leaves_the_flag_alone() -> None:
    events: dict[str, bool] = {}
    watcher = _grasp_watcher(mode="shadow", events=events)
    clock = [0.0]
    watcher.clock = lambda: clock[0]
    watcher.start()
    _offer_widths(watcher, clock, (0.0, 1.0, 0.013), (1.0, 4.0, 0.0))
    assert watcher.grasp_decided_at_s is not None
    assert not watcher.fired and "exit_early" not in events


def test_a_grasp_stage_reads_the_classifier_only_for_the_record() -> None:
    events: dict[str, bool] = {}
    settings = WatcherSettings(
        threshold=0.9, hold_s=0.2, min_s=0.0, mode="fire", rule="grasp", wait_s=9.0
    )
    grip = GripTracker(config.GripConfig(state_names=["right"]), [0])
    watcher = CompletionWatcher(_FakeModel(), settings, events, grip=grip)
    for t in (0.0, 0.2, 0.4):
        watcher.decide(t, 0.99)
    assert watcher.decided_at_s is not None and not watcher.fired


def test_no_state_in_the_observation_stops_the_watcher_without_raising() -> None:
    watcher = _grasp_watcher()
    watcher.start()
    watcher.offer({})
    assert "could not read the grippers" in watcher.error


# --- preflight ----------------------------------------------------------------


def _yaml(tmp_path: Path, stage_four_terminator: str) -> Path:
    text = SMOKE_CONFIG.read_text()
    old = "    terminator:\n      type: timeout\n      timeout_s: 1.0\n    precondition: null\n"
    assert text.count(old) == 2
    path = tmp_path / "classifier.yaml"
    path.write_text(
        text.replace(old, stage_four_terminator + "    precondition: null\n", 1)
    )
    return path


def _parse(path: Path, tmp_path: Path) -> config.StageRunnerConfig:
    return config.parse_config(
        [
            "--config_path",
            str(path),
            f"--dataset.repo_id={REPO_ID}",
            f"--dataset.root={tmp_path / 'dataset'}",
            f"--output.root={tmp_path / 'outputs'}",
        ]
    )


CLASSIFIER_FIRE = (
    "    terminator:\n"
    "      type: classifier\n"
    "      timeout_s: 4.0\n"
    "      model_path: fake://completion\n"
)
GRASP_FIRE = (
    "    terminator:\n"
    "      type: grasp\n"
    "      timeout_s: 4.0\n"
    "      wait_s: 2.0\n"
    "      grip:\n"
    "        state_names: [joint_6.pos]\n"
)


@pytest.mark.parametrize(
    ("terminator", "message"),
    [
        (
            "    terminator:\n      type: classifier\n      model_path: m\n",
            "needs `timeout_s`",
        ),
        (
            "    terminator:\n      type: classifier\n      timeout_s: 4.0\n",
            "needs `model_path`",
        ),
        (CLASSIFIER_FIRE + "      mode: watch\n", "expected one of fire, shadow"),
        (CLASSIFIER_FIRE + "      threshold: 1.5\n", "between 0 and 1"),
        (CLASSIFIER_FIRE + "      min_s: 9.0\n", "could never"),
        (
            "    terminator:\n      type: timeout\n      timeout_s: 4.0\n      model_path: m\n",
            "read only by `terminator.type: classifier` or `grasp`",
        ),
        (
            CLASSIFIER_FIRE + "      grip:\n        state_names: [joint_6.pos]\n",
            "needs `hold_min_m` < `hold_max_m`",
        ),
        (
            CLASSIFIER_FIRE + "      wait_s: 3.0\n",
            "read only by `terminator.type: grasp`",
        ),
        (
            "    terminator:\n      type: grasp\n      timeout_s: 4.0\n",
            "needs a `grip` block",
        ),
        (GRASP_FIRE.replace("      wait_s: 2.0\n", ""), "needs `wait_s` > 0"),
        (GRASP_FIRE.replace("wait_s: 2.0", "wait_s: 5.0"), "could never end"),
        (
            GRASP_FIRE + "        close_below_m: 0.02\n",
            "must be below `open_above_m",
        ),
    ],
)
def test_preflight_refuses_a_malformed_classifier_terminator(
    tmp_path: Path, terminator: str, message: str
) -> None:
    cfg = _parse(_yaml(tmp_path, terminator), tmp_path)
    with pytest.raises(PreflightError, match=message):
        preflight.check_stage_definitions(cfg)


def test_preflight_accepts_a_classifier_terminator(tmp_path: Path) -> None:
    cfg = _parse(_yaml(tmp_path, CLASSIFIER_FIRE), tmp_path)
    preflight.check_stage_definitions(cfg)
    assert cfg.stages[0].terminator.mode == "fire"


def test_preflight_accepts_a_grasp_terminator(tmp_path: Path) -> None:
    cfg = _parse(_yaml(tmp_path, GRASP_FIRE), tmp_path)
    preflight.check_stage_definitions(cfg)
    assert cfg.stages[0].terminator.grip.state_names == ["joint_6.pos"]
    assert cfg.stages[0].terminator.grip.open_above_m == pytest.approx(0.010)


def test_a_gripper_the_robot_lacks_is_an_exit_2(tmp_path: Path) -> None:
    cfg = _parse(_yaml(tmp_path, GRASP_FIRE), tmp_path)
    robot = mock.Mock(observation_features={"joint_0.pos": float})
    with (
        mock.patch.object(completion, "load_completion_models", return_value={}),
        pytest.raises(PreflightError, match="joint_6.pos"),
    ):
        preflight.load_completion_models(cfg, robot)


def test_a_camera_the_robot_lacks_is_an_exit_2(tmp_path: Path) -> None:
    cfg = _parse(_yaml(tmp_path, CLASSIFIER_FIRE), tmp_path)
    robot = mock.Mock(observation_features={"cam_high": (480, 640, 3)})
    model = _FakeModel(keys=("observation.images.cam_low",))
    with (
        mock.patch.object(
            completion, "load_completion_models", return_value={"s4": model}
        ),
        pytest.raises(PreflightError, match="cam_low"),
    ):
        preflight.load_completion_models(cfg, robot)


# --- the chain ----------------------------------------------------------------


def _run(
    tmp_path: Path, terminator: str, model: _FakeModel
) -> tuple[int, list[dict[str, Any]]]:
    path = _yaml(tmp_path, terminator)
    argv = [
        "--config_path",
        str(path),
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.root={tmp_path / 'dataset'}",
        f"--output.root={tmp_path / 'outputs'}",
    ]
    with mock.patch.object(
        preflight, "load_completion_models", return_value={"s4": model}
    ):
        code = cli.main(argv)
    events_path = next((tmp_path / "outputs").glob("*/events.jsonl"))
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    return code, events


def _stage_end(events: list[dict[str, Any]], stage_id: str) -> dict[str, Any]:
    return next(
        e for e in events if e["event"] == "stage_end" and e["stage_id"] == stage_id
    )


def test_a_classifier_ends_the_stage_before_its_timer(tmp_path: Path) -> None:
    model = _FakeModel()
    code, events = _run(tmp_path, CLASSIFIER_FIRE, model)
    assert code == 0
    end = _stage_end(events, "s4")
    assert end["terminator"] == "classifier"
    assert "completion score" in end["reason"]
    # threshold 0.9, hold 0.2 s, min 0.3 s from the stand-in's meta: well inside
    # the 4 s ceiling, with room for a slow CI machine.
    assert end["elapsed_s"] < 3.0
    names = [e["event"] for e in events]
    assert names.index("stage_completion") < names.index("stage_end")
    line = next(e for e in events if e["event"] == "stage_completion")
    assert line["fired"] is True and line["mode"] == "fire"
    assert line["frames_scored"] >= 2 and line["samples"]
    # The next stage still ran, on its own timer.
    assert _stage_end(events, "s5")["terminator"] == "timeout"
    assert model.prepared >= 2


def test_shadow_mode_leaves_the_timer_in_charge(tmp_path: Path) -> None:
    model = _FakeModel()
    shadow = (
        CLASSIFIER_FIRE.replace("timeout_s: 4.0", "timeout_s: 1.5")
        + "      mode: shadow\n"
    )
    code, events = _run(tmp_path, shadow, model)
    assert code == 0
    end = _stage_end(events, "s4")
    assert end["terminator"] == "timeout"
    line = next(e for e in events if e["event"] == "stage_completion")
    assert line["fired"] is False and line["decided_at_s"] is not None
    assert line["decided_at_s"] < 1.5


def test_a_low_score_lets_the_timer_end_the_stage(tmp_path: Path) -> None:
    model = _FakeModel(scores=[0.1] * 100)
    terminator = CLASSIFIER_FIRE.replace("timeout_s: 4.0", "timeout_s: 1.5")
    code, events = _run(tmp_path, terminator, model)
    assert code == 0
    assert _stage_end(events, "s4")["terminator"] == "timeout"
    line = next(e for e in events if e["event"] == "stage_completion")
    assert line["decided_at_s"] is None and line["max_score"] == pytest.approx(0.1)


def test_the_gripper_gate_keeps_a_confident_classifier_from_firing(
    tmp_path: Path,
) -> None:
    # A band no mock joint sits in: the score is 1.0 throughout, the gate never
    # opens, and the timer ends the stage -- the t06 stage-2 case.
    gated = (
        CLASSIFIER_FIRE.replace("timeout_s: 4.0", "timeout_s: 1.5")
        + "      grip:\n"
        + "        state_names: [joint_6.pos, joint_13.pos]\n"
        + "        hold_min_m: 100.0\n"
        + "        hold_max_m: 101.0\n"
    )
    code, events = _run(tmp_path, gated, _FakeModel())
    assert code == 0
    assert _stage_end(events, "s4")["terminator"] == "timeout"
    line = next(e for e in events if e["event"] == "stage_completion")
    assert line["fired"] is False and line["gate_blocked_at_s"] is not None
    assert line["grip_state_names"] == ["joint_6.pos", "joint_13.pos"]
    assert line["grip_samples"] and len(line["grip_samples"][0]) == 3


def test_a_grasp_stage_without_a_grasp_ends_on_its_timer(tmp_path: Path) -> None:
    terminator = GRASP_FIRE.replace("timeout_s: 4.0", "timeout_s: 1.5").replace(
        "wait_s: 2.0", "wait_s: 1.0"
    )
    code, events = _run(tmp_path, terminator, _FakeModel())
    assert code == 0
    assert _stage_end(events, "s4")["terminator"] == "timeout"
    line = next(e for e in events if e["event"] == "stage_completion")
    assert line["rule"] == "grasp" and line["fired"] is False
    assert line["grasp_at_s"] is None and line["grip_samples"]
