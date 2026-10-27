"""The space bar hand-over, driven through lerobot's real record_loop.

No hardware: a fake robot with a base torque switch, a fake leader arm with
align_to, a fake policy that "presses" the space bar at a chosen tick, and a real
LeRobotDataset (no video) for the intervention dataset under a temporary
HF_LEROBOT_HOME.
"""

import logging
import re
from types import SimpleNamespace

import pytest
from fake_policies import (
    ACTION_NAMES,
    FakeDataset,
    FakeRobot,
    IdentityPostprocessor,
    QueuePolicy,
    TickPreprocessor,
)
from lerobot.processor import make_default_processors
from lerobot.scripts import lerobot_record
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot_robot_trossen import base_serial_rearm, intervention_patch, loop_rate_log

# The fixture replaces _on_hub; the Hub test needs the real one.
REAL_ON_HUB = intervention_patch._on_hub

ARM_KEYS = ["left_joint_0.pos", "right_joint_0.pos"]
FOLLOWER_POSE = {"left_joint_0.pos": 0.8, "right_joint_0.pos": -0.6}
LOOP_FPS = 200
TASK = "Move to the tube rack"
# Long enough for several 30 fps reset ticks, so a key can land inside the reset.
RESET_S = 0.2
# The take's limit (episode_time_s); the same length as a default policy episode.
TAKE_S = 0.2


class TorqueRobot(FakeRobot):
    """FakeRobot plus measured arm joints and a base torque switch."""

    def __init__(self, connected_with_torque: bool = True):
        super().__init__()
        self.config = SimpleNamespace(enable_base_motor_torque=connected_with_torque)
        self.torque_calls: list[bool] = []
        self.base.enable_motor_torque = self._enable_motor_torque

    def _enable_motor_torque(self, on: bool):
        self.torque_calls.append(on)
        return True, ""

    def get_observation(self) -> dict:
        return {**super().get_observation(), **FOLLOWER_POSE}


class FakeLeader:
    """A leader arm whose action is its pose plus the base's measured velocity.

    ``keys`` presses a key on the n-th get_action (1-based) of the n-th loop of a
    stage: {("reset", 1, 2): "right"} is the right arrow on the second action of
    the first reset. Stages are "reset" (no dataset) and "record" (the
    intervention dataset), set by ``stage_spy``.
    """

    def __init__(self, keys: dict[tuple[str, int, int], str] | None = None):
        self.pose = {"left_joint_0.pos": 0.0, "right_joint_0.pos": 0.0}
        self.align_calls: list[tuple[dict, float]] = []
        self.keys = keys or {}
        self.events: dict | None = None
        self.stage: str | None = None
        self.stage_loops: dict[str, int] = {}
        self.calls_in_loop = 0
        # (stage, phase, target fps) of the record_loop each get_action ran in.
        self.loops: list[tuple[str | None, str | None, float | None]] = []

    @property
    def action_features(self) -> dict:
        return {name: float for name in ACTION_NAMES}

    def align_to(self, positions: dict, goal_time: float) -> None:
        self.align_calls.append((dict(positions), goal_time))
        self.pose.update(positions)

    def get_action(self) -> dict:
        self.loops.append(
            (
                self.stage,
                loop_rate_log.current_phase(),
                loop_rate_log.current_target_fps(),
            )
        )
        if self.stage is not None:
            self.calls_in_loop += 1
            key = self.keys.get(
                (self.stage, self.stage_loops[self.stage], self.calls_in_loop)
            )
            if key is not None:
                press(self.events, key)
        return {**self.pose, "x.vel": 0.05, "theta.vel": -0.1}


def press(events: dict, key: str) -> None:
    """lerobot's init_keyboard_listener, without the keyboard."""
    if key == "left":
        events["rerecord_episode"] = True
    elif key == "esc":
        events["stop_recording"] = True
    events["exit_early"] = True


def stage_spy(original, leader):
    """Tell the fake leader which kind of record_loop it is being read in."""

    def record_loop(*args, **kwargs):
        if leader is None or kwargs.get("policy") is not None:
            return original(*args, **kwargs)
        stage = "reset" if kwargs.get("dataset") is None else "record"
        leader.stage = stage
        leader.stage_loops[stage] = leader.stage_loops.get(stage, 0) + 1
        leader.calls_in_loop = 0
        try:
            return original(*args, **kwargs)
        finally:
            leader.stage = None

    return record_loop


# record_loop only takes a teleop that is a Teleoperator; anything else makes its
# teleop branch log "No policy or teleoperator provided" and spin without ever
# advancing its clock.
Teleoperator.register(FakeLeader)


class SpacePolicy(QueuePolicy):
    """Presses the space bar when it is asked for its ``press_at``-th action."""

    def __init__(self, press_at: int):
        super().__init__(action_steps=10)
        self.press_at = press_at
        self.asked = 0

    def select_action(self, batch):
        self.asked += 1
        if self.asked == self.press_at:
            intervention_patch._on_space()
        return super().select_action(batch)


@pytest.fixture
def patch_env(tmp_path, monkeypatch):
    from lerobot.datasets import lerobot_dataset
    from lerobot.utils import constants

    monkeypatch.setattr(constants, "HF_LEROBOT_HOME", tmp_path)
    monkeypatch.setattr(lerobot_dataset, "HF_LEROBOT_HOME", tmp_path)
    monkeypatch.setattr(intervention_patch, "REPO_ID", "local/intervention_task01")
    monkeypatch.setattr(intervention_patch, "_on_hub", lambda repo_id: False)
    monkeypatch.setattr(intervention_patch, "_listener_started", True)
    monkeypatch.setattr(intervention_patch, "_space_at", None)
    monkeypatch.setattr(intervention_patch, "_dataset", None)
    monkeypatch.setattr(intervention_patch, "_saved_this_run", 0)
    monkeypatch.setattr(intervention_patch, "_finalized", False)
    # No real align delay in tests.
    monkeypatch.setattr(intervention_patch, "LEADER_ALIGN_MIN_GOAL_TIME_S", 0.0)
    # A short reset that ends by itself unless a test presses a key in it.
    monkeypatch.setattr(intervention_patch, "reset_time_s", lambda: RESET_S)
    monkeypatch.setattr(intervention_patch, "episode_time_s", lambda: TAKE_S)
    return tmp_path


def run_policy_episode(robot, leader, policy, events, ticks: int = 40):
    teleop_processor, robot_processor, observation_processor = make_default_processors()
    # As in a real eval, the eval dataset and the loop share --dataset.fps.
    eval_dataset = FakeDataset(fps=LOOP_FPS, repository_id="local/eval_task01")
    # Inside loop_rate_log's phase tagging, as installed in a real run.
    wrapped = intervention_patch._wrap_record_loop(
        loop_rate_log._wrap_record_loop(stage_spy(lerobot_record.record_loop, leader))
    )
    if leader is not None:
        leader.events = events
    wrapped(
        robot=robot,
        events=events,
        fps=LOOP_FPS,
        teleop_action_processor=teleop_processor,
        robot_action_processor=robot_processor,
        robot_observation_processor=observation_processor,
        teleop=leader,
        dataset=eval_dataset,
        policy=policy,
        preprocessor=TickPreprocessor(),
        postprocessor=IdentityPostprocessor(),
        control_time_s=ticks / LOOP_FPS,
        single_task=TASK,
    )
    return eval_dataset


def new_events() -> dict:
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


def test_space_bar_hands_over_and_records(patch_env):
    robot, leader, events = TorqueRobot(), FakeLeader(), new_events()
    eval_dataset = run_policy_episode(robot, leader, SpacePolicy(press_at=5), events)

    # The policy stopped where the space bar was pressed.
    assert 3 <= len(eval_dataset.frames) <= 6
    # The leader went to the follower's measured pose before the hand-over.
    assert leader.align_calls and leader.align_calls[0][0] == FOLLOWER_POSE
    # Torque off for the intervention, back on afterwards.
    assert robot.torque_calls == [False, True]
    assert not events["stop_recording"]

    dataset = intervention_patch._dataset
    assert dataset is not None
    assert dataset.fps == intervention_patch.DATASET_FPS == 30
    assert dataset.num_episodes == 1
    assert intervention_patch._saved_this_run == 1

    dataset.finalize()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Read back from disk, as training would.
    dataset = LeRobotDataset(intervention_patch.REPO_ID)
    assert dataset.fps == 30
    frames = dataset.hf_dataset
    count = len(frames)
    assert count > 3
    # Nominal 30 fps grid, as in the acquisition datasets.
    timestamps = [float(t) for t in frames["timestamp"]]
    assert timestamps[:3] == pytest.approx([0.0, 1 / 30, 2 / 30])
    # The recorded action is the leader's pose plus the measured base velocity.
    first_action = [float(v) for v in frames["action"][0]]
    names = dataset.features["action"]["names"]
    recorded = dict(zip(names, first_action))
    assert recorded["left_joint_0.pos"] == pytest.approx(0.8)
    assert recorded["theta.vel"] == pytest.approx(-0.1)
    assert dataset.meta.tasks.index.tolist() == [TASK]


def test_intervention_loop_runs_as_acquisition(patch_env):
    robot, leader, events = TorqueRobot(), FakeLeader(), new_events()
    run_policy_episode(robot, leader, SpacePolicy(press_at=5), events)

    # The leader is read inside a loop only during the intervention.
    loops = {(phase, fps) for _, phase, fps in leader.loops if phase is not None}
    # Teleop loops at lerobot-record's default fps, whatever --dataset.fps says.
    assert loops == {("teleop", 30.0)}
    # So the base keeps the driver's receive wait, as it had in acquisition.
    assert base_serial_rearm.loop_refusal_reason(*loops.pop()) is not None


def test_a_reset_comes_before_the_recording(patch_env):
    robot, leader, events = TorqueRobot(), FakeLeader(), new_events()
    run_policy_episode(robot, leader, SpacePolicy(press_at=5), events)

    stages = [stage for stage, _, _ in leader.loops if stage is not None]
    # Not recorded first, then recorded -- as a reset and an episode in acquisition.
    assert stages[0] == "reset" and stages[-1] == "record"
    assert stages == sorted(stages, key=["reset", "record"].index)
    # The reset ended by its time limit and recording started by itself.
    assert leader.stage_loops == {"reset": 1, "record": 1}
    assert intervention_patch._dataset.num_episodes == 1
    # Base torque was off from the reset on.
    assert robot.torque_calls == [False, True]


def test_right_arrow_in_the_reset_starts_recording(patch_env):
    robot, events = TorqueRobot(), new_events()
    leader = FakeLeader(keys={("reset", 1, 1): "right"})
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), events)

    reset_calls = [s for s, _, _ in leader.loops if s == "reset"]
    assert len(reset_calls) == 1  # ended at once, not by the time limit
    assert intervention_patch._dataset.num_episodes == 1
    assert not any(events.values())


def test_left_arrow_in_the_first_reset_only_starts_recording(patch_env):
    robot, events = TorqueRobot(), new_events()
    leader = FakeLeader(keys={("reset", 1, 1): "left"})
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), events)

    # One take, kept: the left arrow must not throw away the take that follows.
    assert leader.stage_loops == {"reset": 1, "record": 1}
    assert intervention_patch._dataset.num_episodes == 1
    # Not passed on: lerobot-record would throw the eval episode away.
    assert not events["rerecord_episode"]
    assert not events["stop_recording"]


def test_esc_in_the_first_reset_stops_with_nothing_recorded(patch_env):
    robot, events = TorqueRobot(), new_events()
    leader = FakeLeader(keys={("reset", 1, 1): "esc"})
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), events)

    assert "record" not in leader.stage_loops
    assert intervention_patch._dataset.num_episodes == 0
    assert intervention_patch._saved_this_run == 0
    assert events["stop_recording"]
    assert robot.torque_calls == [False, True]  # torque back on anyway


def test_left_arrow_discards_and_records_again(patch_env):
    robot, events = TorqueRobot(), new_events()
    leader = FakeLeader(keys={("record", 1, 2): "left"})
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), events)

    # The left arrow went to a reset, then a second take was recorded.
    assert leader.stage_loops == {"reset": 2, "record": 2}
    dataset = intervention_patch._dataset
    assert dataset.num_episodes == 1  # the discarded take is not saved
    assert not events["rerecord_episode"]  # the eval episode is not re-recorded
    assert not events["stop_recording"]


def test_esc_in_the_reset_after_a_left_arrow_discards_and_stops(patch_env):
    robot, events = TorqueRobot(), new_events()
    leader = FakeLeader(keys={("record", 1, 2): "left", ("reset", 2, 1): "esc"})
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), events)

    assert leader.stage_loops == {"reset": 2, "record": 1}
    assert intervention_patch._dataset.num_episodes == 0
    assert events["stop_recording"]


def test_esc_while_recording_saves_and_stops(patch_env):
    robot, events = TorqueRobot(), new_events()
    leader = FakeLeader(keys={("record", 1, 2): "esc"})
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), events)

    assert leader.stage_loops == {"reset": 1, "record": 1}
    assert intervention_patch._dataset.num_episodes == 1
    assert events["stop_recording"]


def test_reset_time_comes_from_the_command_line():
    assert intervention_patch.reset_time_s(["--dataset.reset_time_s=90"]) == 90.0
    assert intervention_patch.reset_time_s(["--dataset.reset_time_s", "12.5"]) == 12.5
    # Not given: lerobot-record's own default.
    from lerobot.scripts.lerobot_record import DatasetRecordConfig

    default = DatasetRecordConfig.__dataclass_fields__["reset_time_s"].default
    assert intervention_patch.reset_time_s(["--dataset.fps=21"]) == float(default)


def test_take_time_comes_from_the_command_line():
    assert intervention_patch.episode_time_s(["--dataset.episode_time_s=120"]) == 120.0
    # stage_runner passes no --dataset.* flags: lerobot-record's own default.
    from lerobot.scripts.lerobot_record import DatasetRecordConfig

    default = DatasetRecordConfig.__dataclass_fields__["episode_time_s"].default
    take_s = intervention_patch.episode_time_s(["--config_path", "x.yaml"])
    assert take_s == float(default)


def test_take_limit_is_not_the_callers_control_time(patch_env, monkeypatch):
    # stage_runner calls record_loop with the stage's timer; the take must not
    # inherit it. A take limit four times the policy episode records more frames
    # than the episode's own limit would allow.
    monkeypatch.setattr(intervention_patch, "episode_time_s", lambda: 4 * TAKE_S)
    robot, leader, events = TorqueRobot(), FakeLeader(), new_events()
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), events, ticks=40)

    frames = intervention_patch._dataset.num_frames
    assert frames > 40 * intervention_patch.DATASET_FPS / LOOP_FPS * 2


def test_hand_over_timing_is_logged(patch_env, caplog):
    robot, leader, events = TorqueRobot(), FakeLeader(), new_events()
    with caplog.at_level(logging.WARNING, logger=intervention_patch.logger.name):
        run_policy_episode(robot, leader, SpacePolicy(press_at=5), events)

    timing = [
        r.getMessage()
        for r in caplog.records
        if "after the space bar" in r.getMessage()
    ]
    assert len(timing) == 1
    match = re.search(r"stopped in (\d+) ms and .* went out at (\d+) ms", timing[0])
    assert match
    stopped_ms, zeroed_ms = (int(v) for v in match.groups())
    # The zero command follows the loop exit, and a 200 Hz fake loop exits fast.
    assert 0 <= stopped_ms <= zeroed_ms < 1000
    # Cleared for the next policy episode.
    assert intervention_patch._space_at is None


def test_no_leader_stops_the_run_and_records_nothing(patch_env):
    robot, events = TorqueRobot(), new_events()
    run_policy_episode(robot, None, SpacePolicy(press_at=3), events)

    assert events["stop_recording"]
    assert intervention_patch._dataset is None
    assert robot.torque_calls == []
    assert robot.base_stops >= 1  # the base was still zeroed


def test_base_left_alone_when_connected_without_torque(patch_env):
    robot, leader, events = (
        TorqueRobot(connected_with_torque=False),
        FakeLeader(),
        new_events(),
    )
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), events)

    assert robot.torque_calls == []
    assert intervention_patch._dataset.num_episodes == 1


def test_no_space_bar_changes_nothing(patch_env):
    robot, leader, events = TorqueRobot(), FakeLeader(), new_events()
    eval_dataset = run_policy_episode(
        robot, leader, SpacePolicy(press_at=10_000), events, ticks=20
    )

    assert len(eval_dataset.frames) == 20
    assert leader.align_calls == []
    assert intervention_patch._dataset is None


def test_space_bar_outside_a_policy_episode_is_ignored(patch_env):
    intervention_patch._on_space()
    assert intervention_patch._requested is False


def test_appends_to_an_existing_dataset(patch_env):
    robot, leader = TorqueRobot(), FakeLeader()
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), new_events())
    intervention_patch._dataset.finalize()

    # A new run: the module state starts over, the dataset is on disk.
    intervention_patch._dataset = None
    intervention_patch._saved_this_run = 0
    run_policy_episode(
        TorqueRobot(), FakeLeader(), SpacePolicy(press_at=3), new_events()
    )
    assert intervention_patch._dataset.num_episodes == 2


def test_a_failure_before_the_take_leaves_the_base_torque_on(patch_env, monkeypatch):
    # The take and reset limits are read before the torque goes off: a failure
    # there must not leave an unpowered base for the next base stage.
    def broken() -> float:
        raise RuntimeError("no DatasetRecordConfig")

    monkeypatch.setattr(intervention_patch, "episode_time_s", broken)
    robot, leader, events = TorqueRobot(), FakeLeader(), new_events()
    with pytest.raises(RuntimeError):
        run_policy_episode(robot, leader, SpacePolicy(press_at=3), events)

    assert robot.torque_calls == []


def test_a_hub_that_cannot_be_asked_stops_the_run(patch_env, monkeypatch):
    # No local copy and no answer from the Hub: starting a new dataset at
    # episode 0 would let its push write over the one on the Hub.
    from huggingface_hub import HfApi

    def unreachable(self, *args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(HfApi, "repo_exists", unreachable)
    monkeypatch.setattr(intervention_patch, "_on_hub", REAL_ON_HUB)
    robot, leader, events = TorqueRobot(), FakeLeader(), new_events()
    run_policy_episode(robot, leader, SpacePolicy(press_at=3), events)

    assert events["stop_recording"]
    assert intervention_patch._dataset is None
    assert not (patch_env / intervention_patch.REPO_ID).exists()
    assert robot.torque_calls == []


def test_an_empty_dataset_from_an_earlier_run_is_set_aside(patch_env):
    # ESC in the first reset: the dataset was created but nothing was saved, so
    # only its meta/info.json is on disk.
    leader = FakeLeader(keys={("reset", 1, 1): "esc"})
    run_policy_episode(TorqueRobot(), leader, SpacePolicy(press_at=3), new_events())
    assert intervention_patch._dataset.num_episodes == 0

    # The next run records into a fresh dataset instead of stopping.
    intervention_patch._dataset = None
    intervention_patch._saved_this_run = 0
    events = new_events()
    run_policy_episode(TorqueRobot(), FakeLeader(), SpacePolicy(press_at=3), events)

    assert not events["stop_recording"]
    assert intervention_patch._dataset.num_episodes == 1
    owner, name = intervention_patch.REPO_ID.split("/")
    assert list((patch_env / owner).glob(f"{name}.empty-*"))


def test_recordable_features_drop_defaults_and_video_info():
    features = {
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
            "info": {"video.fps": 21},
        },
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    out = intervention_patch.recordable_features(features)
    assert "timestamp" not in out
    assert "info" not in out["observation.images.cam_high"]
    assert out["action"] == features["action"]
    assert "info" in features["observation.images.cam_high"]  # input untouched


def test_align_goal_time_ignores_grippers_and_has_a_floor(monkeypatch):
    monkeypatch.setattr(intervention_patch, "LEADER_ALIGN_MIN_GOAL_TIME_S", 2.0)
    start = {"left_joint_0.pos": 0.0, "left_left_carriage_joint.pos": 0.0}
    assert intervention_patch.align_goal_time(start, {"left_joint_0.pos": 0.1}) == 2.0
    assert intervention_patch.align_goal_time(
        start, {"left_joint_0.pos": 1.5}
    ) == pytest.approx(3.0)
    assert (
        intervention_patch.align_goal_time(start, {"left_left_carriage_joint.pos": 9.0})
        == 2.0
    )
