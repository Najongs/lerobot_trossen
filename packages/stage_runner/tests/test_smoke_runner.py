"""Smoke path for the runner body: the whole chain with no hardware, no network.

Drives cli.main against tests/configs/smoke_mock.yaml, which selects our own mock
robot and mock policies through lerobot's real factories, so lerobot's REAL
record_loop executes. There is deliberately no hand-written fake loop: a second
implementation of the one seam this package exists to funnel would drift from the
real one silently, and the mock robot plus the mock policies already make the real
loop runnable with nothing plugged in.

Covers the confirmed six-step acceptance path -- YAML parse, policy preload, stage
transition, base zero, one continuous episode, disconnect -- plus both designated
verification targets:

  A. draccus eats our dataclass with a nested `stages:` list read from a file.
  B. a task string that changes per stage inside ONE episode is recorded per frame.

Both were answered by direct evidence before this test existed (draccus 0.10.0 run
by hand; lerobot_dataset.py:1250-1261 read, where save_episode does
`episode_tasks = list(set(tasks))` and writes a per-frame task_index), so these
assertions are regression guards rather than discoveries.

What this path does NOT exercise: the mock robot has no cameras, so
image_writer_threads computes to 0 and the image writer, the video encoding and
VideoEncodingManager's encoder flush never run, and none of the real robot's
defenses (velocity pacing, e-stop checks, camera latency) are touched. A
camera-shaped or pacing-shaped failure will first appear on hardware.

THIS TEST IS NOT EVIDENCE ABOUT LOOP RATE, and nothing it asserts or prints may
be read as such. A mock robot that answers get_observation from a dict keeps a
clean 30 Hz by construction; the real loop pays three RealSense reads, ACT
inference and PNG encoding per frame, and another session measured 20.4 Hz on
it. That is why no assertion here touches `hertz` or elapsed_s beyond "frames
were written at all": a green run says the plumbing is connected, not that the
30 fps the YAML timers assume holds. The one number that settles it is the
per-stage `hertz` in events.jsonl from the FIRST hardware run (spec: MANDATORY
FIRST-RUN PROCEDURE), read before the 12.9 / 20.8 timeouts are fixed.

Below the happy path this file also drives the SAFETY branches -- a dead camera,
an executor error, a Ctrl+C, a failing emit, an abort, and both stop paths
failing at once. Build item 18 says "the runner body gets exactly ONE smoke
path", and those cases are a deliberate deviation from it: the round-2 review
returned DO-NOT-SHIP on three safety fixes that had no test at all, and a fix
whose only evidence is an ad-hoc probe is one tidy-up away from being reverted
with the suite still green. Each of them was checked by reverting the fix it
guards and watching it fail.

    uv run pytest packages/stage_runner/tests/test_smoke_runner.py
    uv run python packages/stage_runner/tests/test_smoke_runner.py   # no pytest
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd

from lerobot.robots import Robot
from lerobot.robots.config import RobotConfig

from stage_runner import aggregate, cli, config, policies, record_adapter, transitions
from stage_runner import events as event_stream
from stage_runner.events import BASE_FIELD_NAMES, EVENT_NAMES

CONFIG_PATH = Path(__file__).parent / "configs" / "smoke_mock.yaml"
REPO_ID = "local/eval_smoke_chain"
STAGE_FOUR_TASK = "mock stage four"
STAGE_FIVE_TASK = "mock stage five"

# N stages -> N transition events, not N-1: since 2026-09-08 the boundary after
# the LAST stage gets a line too, carrying to_stage_id=null. That line is the
# only place a stop_base outcome reaches disk, and the last boundary is the one
# guarding the seconds-long save_episode / encoder flush -- suppressing it made a
# run whose final base stop silently failed byte-identical to a clean one.
EXPECTED_EVENT_SEQUENCE = (
    "trial_start",
    "stage_start",
    "stage_end",
    "transition",
    "stage_start",
    "stage_end",
    "transition",
    "trial_end",
)


def test_smoke_chain_runs_without_hardware(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "outputs"
    argv = [
        "--config_path",
        str(CONFIG_PATH),
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.root={dataset_root}",
        f"--output.root={output_root}",
    ]

    # (a) VERIFICATION TARGET A. Parsing here rather than reading cli.main's
    # internals: the claim under test is that draccus alone turns this file plus
    # one CLI token into our dataclass tree. No register_plugins() first, because
    # `stage_runner_mock_robot` is registered by parse_config importing
    # stage_runner.mock_robot itself -- the third-party plugin scan is only what
    # `mobileai_robot` needs. (That import used to ride on the package __init__,
    # which is now lazy so stage_runner.aggregate stays stdlib-only.)
    parsed = config.parse_config(argv)
    assert parsed.version == 1, "version: 1 은 스키마 판 식별용 / must survive parsing"
    assert [stage.id for stage in parsed.stages] == ["s4", "s5"], (
        "중첩 stages: 리스트가 순서대로 파싱돼야 한다"
    )
    # The nested terminator MAPPING parses, and its unset fields fill from the
    # dataclass defaults rather than staying None.
    assert parsed.stages[0].terminator.type == "timeout"
    assert parsed.stages[0].terminator.timeout_s == 1.0
    assert parsed.stages[0].instruction == STAGE_FOUR_TASK
    assert parsed.stages[1].instruction == STAGE_FIVE_TASK
    # The ChoiceRegistry field resolved from the nested `robot:` block's `type:`.
    assert type(parsed.robot).__name__ == "MockRobotConfig"
    # The one CLI override reached the field it names.
    assert parsed.dataset.repo_id == REPO_ID
    assert parsed.output.root == str(output_root)

    # cli.main constructs the robot and never hands it back, so wrap the one
    # factory call it makes. This is test instrumentation, not an injection seam:
    # production code has no branch for it.
    created_robots: list[Robot] = []
    original_make_robot = record_adapter.make_robot

    def _capturing_make_robot(robot_config: RobotConfig) -> Robot:
        robot = original_make_robot(robot_config)
        created_robots.append(robot)
        return robot

    record_adapter.make_robot = _capturing_make_robot
    try:
        exit_code = cli.main(argv)
    finally:
        record_adapter.make_robot = original_make_robot

    assert exit_code == cli.EXIT_OK, f"cli.main returned {exit_code}"
    assert len(created_robots) == 1

    # (f) Teardown ran. robot.disconnect() is in cli.main's finally, and it is
    # the only thing that stops the base on a normal exit.
    robot = created_robots[0]
    assert robot.is_connected is False, "finally 블록이 disconnect를 안 불렀다"

    run_directories = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert len(run_directories) == 1, (
        f"expected one run directory, got {run_directories}"
    )
    run_directory = run_directories[0]

    # (e) Both snapshots, because neither alone answers both "what did I edit"
    # (the source file, comments included) and "what actually ran" (the resolved
    # config, CLI override included).
    source_snapshot = run_directory / "config.source.yaml"
    resolved_snapshot = run_directory / "config.resolved.yaml"
    assert source_snapshot.exists(), "config.source.yaml 이 없다"
    assert resolved_snapshot.exists(), "config.resolved.yaml 이 없다"
    assert source_snapshot.read_text() == CONFIG_PATH.read_text(), (
        "source 스냅샷은 축자 사본이어야 한다 (주석이 rad 가정·p95 출처를 나른다)"
    )
    assert REPO_ID in resolved_snapshot.read_text(), (
        "resolved 스냅샷이 --dataset.repo_id 오버라이드를 안 담았다"
    )

    # (b) The event stream, in order and complete.
    events = [
        json.loads(line)
        for line in (run_directory / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [event["event"] for event in events] == list(EXPECTED_EVENT_SEQUENCE)
    for event in events:
        assert event["event"] in EVENT_NAMES
        missing = [name for name in BASE_FIELD_NAMES if name not in event]
        assert not missing, f"{event['event']} 에 기본 필드 누락: {missing}"
        assert event["schema_version"] == 1
    # trial_start carries the three WRITER identities, and this asserts they are
    # present and non-empty on every run. They are what tells two pooled trials
    # apart when the writer changed between them -- a metric silently changing
    # meaning is worse than a metric that is missing -- and aggregate.heterogeneity
    # gets a column per field, so a run written without them would be pooled as
    # "same as everything else" instead of flagged. This assertion is the writer
    # half of that pair and the reason the reader half can rely on the fields.
    trial_start = events[0]
    assert trial_start["event"] == "trial_start"
    for field in ("stage_runner_version", "lerobot_version", "config_version"):
        assert trial_start[field], f"trial_start에 {field} 가 비었다"
    assert trial_start["config_version"] == parsed.version

    # Each policy descriptor says whether its warm-up pass ran, because that is
    # what decides whether the stage's `hertz` includes first-inference CUDA /
    # cuDNN init. Mock bundles are never warmed, so false here is the truth and
    # not a regression -- the True case is asserted in
    # test_load_bundles_records_whether_the_warm_up_ran, which does not need a
    # checkpoint.
    assert [entry["stage_id"] for entry in trial_start["policies"]] == ["s4", "s5"]
    assert all(entry["warmed_up"] is False for entry in trial_start["policies"]), (
        trial_start["policies"]
    )

    transitions = [event for event in events if event["event"] == "transition"]
    assert len(transitions) == 2, "N개 단계면 전환 이벤트도 N개 (마지막 경계 포함)"
    # The chain crossed once, so exactly one line names a next stage; the last
    # boundary leads nowhere and says so instead of being absent.
    assert [event["to_stage_id"] for event in transitions] == ["s5", None]
    assert [event["from_stage_id"] for event in transitions] == ["s4", "s5"]
    # A clean run took the primary path at every boundary. If this ever reads
    # "fallback" or "failed" on the mock, build_hold_action broke: the mock has
    # no .base, so its fallback is a no-op and would report "failed".
    assert all(event["stop_base_path"] == "primary" for event in transitions)
    assert all(event["reason"] == "stop_base" for event in transitions)

    stage_ends = [event for event in events if event["event"] == "stage_end"]
    assert [event["stage_id"] for event in stage_ends] == ["s4", "s5"]
    # Both stages hit their 1.0 s timeout; nothing pressed a key.
    assert all(event["terminator"] == "timeout" for event in stage_ends)
    # Frames were written at all -- deliberately NOT how many, and nothing here
    # asserts `hertz` or elapsed_s. A cameraless mock keeps 30 Hz by
    # construction, so any rate this run reports (including the runner's own
    # "30 frames in 1.00s" log line) is an artifact of the mock and must not be
    # quoted as evidence about the real loop.
    assert all(event["frames"] > 0 for event in stage_ends), "프레임이 안 쌓였다"
    assert events[-1]["completed"] is True
    assert events[-1]["stage_count"] == 2

    # (c) The transition action, asserted as a WHOLE dict. "some .pos key is
    # present" would stay green if build_hold_action filled the arm keys with
    # 0.0 instead of the present position -- the single most likely regression
    # in that function, and one that on hardware slews both WidowXAI arms to the
    # zero pose at every stage boundary. mock_robot seeds each joint to a
    # different non-zero value precisely so that bug is detectable here.
    stop_action = transitions[0]["stop_action"]
    assert stop_action["x.vel"] == 0.0
    assert stop_action["theta.vel"] == 0.0
    # Full width: a base-only dict raises KeyError at widowxai_follower.py:272,
    # which iterates every configured joint name.
    assert set(stop_action) == set(robot.action_features), stop_action
    position_keys = sorted(key for key in stop_action if key.endswith(".pos"))
    assert position_keys == sorted(
        f"joint_{index}.pos" for index in range(robot.config.joint_count)
    ), position_keys
    # The arms are position-controlled, so a correct hold repeats the pose the
    # previous frame left them in, joint for joint. sent_actions[boundary] is
    # the stop_base send itself (boundary frames were recorded before it), and
    # sent_actions[boundary - 1] is the last command of s4.
    boundary = int(stage_ends[0]["frame_idx"])
    assert stop_action == robot.sent_actions[boundary], (
        "이벤트의 stop_action이 실제로 보낸 것과 다르다"
    )
    held = {key: value for key, value in stop_action.items() if key.endswith(".pos")}
    previous = {
        key: value
        for key, value in robot.sent_actions[boundary - 1].items()
        if key.endswith(".pos")
    }
    assert held == previous, (held, previous)
    assert all(value != 0.0 for value in held.values()), (
        "모든 관절이 0.0 -- 현재 위치가 아니라 영점을 명령했다 / a hold that zeroes "
        "the arms instead of reading the present position"
    )
    # The timing field is stop_base_s, NOT gap_s: it covers one get_observation
    # plus one send_action, and aggregate.py derives the real inter-stage gap
    # from the stage_end -> stage_start wall clocks instead. Asserted so the
    # writer and that reader cannot drift apart silently.
    assert "gap_s" not in transitions[0], transitions[0]
    assert isinstance(transitions[0]["stop_base_s"], float), transitions[0]

    # (d) ONE episode for the whole chain, not one per stage.
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1, "체인 전체가 에피소드 하나여야 한다"

    # (d) VERIFICATION TARGET B. The per-frame task changes at the stage
    # boundary inside that one episode.
    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    task_by_index = {int(row.task_index): str(task) for task, row in tasks.iterrows()}
    assert set(task_by_index.values()) == {STAGE_FOUR_TASK, STAGE_FIVE_TASK}

    frames = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted(dataset_root.glob("data/**/*.parquet"))
        ]
    ).sort_values("frame_index")
    frame_tasks = [task_by_index[int(index)] for index in frames["task_index"]]
    assert len(set(frame_tasks)) == 2, "한 에피소드 안에서 task 문자열이 안 바뀌었다"
    # The switch lands exactly where s4 ended: stage_end carries the buffered
    # frame count at emit time, so every frame before it belongs to s4.
    boundary = stage_ends[0]["frame_idx"]
    assert frame_tasks[: int(boundary)] == [STAGE_FOUR_TASK] * int(boundary)
    assert set(frame_tasks[int(boundary) :]) == {STAGE_FIVE_TASK}

    # (g) THE READER PARSES WHAT THE WRITER JUST WROTE. aggregate.py pins the
    # wire format independently (it is stdlib-only, so it cannot import the
    # writer's constants), which means the two can drift with both files' own
    # tests still green -- test_aggregate.py reads a frozen fixture, and
    # everything above here reads the raw JSON. This is the one assertion that
    # joins them, and it is where the 2026-09-08 schema change would have shown
    # up: N stages now produce N boundaries, not N-1.
    trial = aggregate.build_trial(aggregate.load_events(run_directory / "events.jsonl"))
    assert trial.completed is True
    assert [stage.stage_id for stage in trial.stages] == ["s4", "s5"]
    assert len(trial.stop_base_durations_s) == 2, trial.stop_base_durations_s
    assert trial.stop_base_fallback_stage_ids == ()
    assert trial.stop_base_failed_stage_ids == ()
    # One CROSSING for two stages: the second boundary leads nowhere, so it
    # contributes a stop_base duration and no inter-stage gap.
    assert len(trial.transition_gaps_s) == 1, trial.transition_gaps_s


# ---------------------------------------------------------------------------
# The safety paths. Everything above this line is the ONE happy chain; what
# follows are the branches the round-2 reviewers returned DO-NOT-SHIP on -- the
# base stop when the camera is dead, the base stop when the exception came from
# somewhere other than the executor, and the abort. They were verified by
# ad-hoc probes when they were written, and a probe is not a guard: the failure
# they protect against is someone tidying `except BaseException` back to
# `except Exception`, or moving the stop_base call below the stage_end emit,
# with every test still green and the first symptom on hardware being a base
# that drove through a Ctrl+C.
#
# All of them drive the REAL cli.main against the same mock config. The one
# measurement each makes is the base velocity AT THE ENTRY TO THE ENCODER
# FLUSH (VideoEncodingManager.__exit__): that window is the image-writer drain
# plus the ffmpeg flush, seconds to minutes on hardware, and it is the last
# moment before robot.disconnect() would zero the base anyway and mask
# everything.

DRIVING_BASE = {"x.vel": 0.4, "theta.vel": 0.0}
STOPPED_BASE = {"x.vel": 0.0, "theta.vel": 0.0}


class _FakeBase:
    """Stand-in for MobileAIRobot.base (a TrossenSlate), the fallback's target.

    MockRobot deliberately has NO .base -- it models a robot where the fallback
    is a no-op -- so the fallback path cannot be exercised without one. This is
    the smallest thing that behaves like the real driver where the fallback
    touches it: set_cmd_vel returns a bool, which mobileai.py tests with
    ``if not self.base.set_cmd_vel(...)`` at :554 and :569, and which is a real
    delivery signal unlike send_action's echo.
    """

    def __init__(self, robot: Any, accepts: bool = True) -> None:
        self.robot = robot
        self.accepts = accepts
        self.commands: list[tuple[float, float]] = []

    def set_cmd_vel(self, x_vel: float, theta_vel: float) -> bool:
        self.commands.append((float(x_vel), float(theta_vel)))
        if not self.accepts:
            return False
        self.robot._base_velocity = {
            "x.vel": float(x_vel),
            "theta.vel": float(theta_vel),
        }
        return True


class _Faults:
    """Switches the hooks flip mid-run, so no case counts frames.

    Deriving "the camera dies during stage 2" from an observation-read COUNT
    would tie the test to how many frames a 1.0 s stage happens to record on the
    machine running it. The stage hooks below fire on stage boundaries instead,
    which is what the cases are actually about.
    """

    def __init__(self) -> None:
        self.observation_fails = False
        # A Ctrl+C landing inside the observation read, which is where the
        # operator's reflex stop lands most often: build_hold_action's
        # get_observation is the longest call in a stage boundary.
        self.observation_interrupts = False


def _instrument(
    robot: Any, faults: _Faults, *, with_base: bool, base_accepts: bool
) -> Any:
    if with_base:
        robot.base = _FakeBase(robot, accepts=base_accepts)
    original_get_observation = robot.get_observation

    def get_observation() -> dict[str, Any]:
        if faults.observation_interrupts:
            raise KeyboardInterrupt
        if faults.observation_fails:
            # The measured fault this whole path exists for: MobileAIRobot
            # reads all three RealSense cameras through an unguarded async_read
            # (mobileai.py:526-528), so a dead camera takes out get_observation
            # -- and with it the hold action that the primary stop_base needs.
            raise RuntimeError("RealSense async_read timed out (injected)")
        return original_get_observation()

    robot.get_observation = get_observation
    return robot


def _run_chain(
    tmp_path: Path,
    *,
    faults: _Faults | None = None,
    with_base: bool = True,
    base_accepts: bool = True,
    before_stage: Any = None,
    after_stage: Any = None,
    preset_events: dict[str, bool] | None = None,
    emit_patch: Any = None,
) -> dict[str, Any]:
    """Drive cli.main once with a fault injected and collect what it left behind.

    Patches only the four seams record_adapter already exposes as module-level
    functions, which cli and executors call through the module. Nothing in
    production code branches on any of this.
    """
    faults = _Faults() if faults is None else faults
    argv = [
        "--config_path",
        str(CONFIG_PATH),
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.root={tmp_path / 'dataset'}",
        f"--output.root={tmp_path / 'outputs'}",
    ]
    captured: dict[str, Any] = {"robot": None, "base_at_encoder_flush": None}

    real_make_robot = record_adapter.make_robot

    def make_robot(robot_config: RobotConfig) -> Robot:
        robot = _instrument(
            real_make_robot(robot_config),
            faults,
            with_base=with_base,
            base_accepts=base_accepts,
        )
        captured["robot"] = robot
        return robot

    real_manager = record_adapter.video_encoding_manager

    @contextlib.contextmanager
    def video_encoding_manager(dataset: Any) -> Any:
        with real_manager(dataset):
            try:
                yield
            finally:
                # __exit__ has not run yet: this is the instant the flush
                # begins, and the last one at which a still-driving base is
                # this package's fault rather than disconnect()'s to clean up.
                captured["base_at_encoder_flush"] = dict(
                    captured["robot"]._base_velocity
                )

    real_call = record_adapter.call_record_loop
    calls = {"n": 0}

    def call_record_loop(**kwargs: Any) -> Any:
        calls["n"] += 1
        if before_stage is not None:
            before_stage(calls["n"], captured)
        result = real_call(**kwargs)
        if after_stage is not None:
            after_stage(calls["n"], captured)
        return result

    real_events = record_adapter.make_keyboard_events

    def make_keyboard_events() -> tuple[Any | None, dict[str, bool]]:
        listener, keyboard_events = real_events()
        if preset_events:
            keyboard_events.update(preset_events)
        return listener, keyboard_events

    patches = [
        mock.patch.object(record_adapter, "make_robot", make_robot),
        mock.patch.object(record_adapter, "call_record_loop", call_record_loop),
        mock.patch.object(
            record_adapter, "video_encoding_manager", video_encoding_manager
        ),
        mock.patch.object(record_adapter, "make_keyboard_events", make_keyboard_events),
    ]
    if emit_patch is not None:
        patches.append(emit_patch)

    raised: BaseException | None = None
    exit_code: int | None = None
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        try:
            exit_code = cli.main(argv)
        except BaseException as error:  # noqa: BLE001 -- the case under test
            raised = error

    events_path = next((tmp_path / "outputs").glob("*/events.jsonl"))
    return {
        "exit_code": exit_code,
        "raised": raised,
        "robot": captured["robot"],
        "base_at_encoder_flush": captured["base_at_encoder_flush"],
        "events": [
            json.loads(line)
            for line in events_path.read_text().splitlines()
            if line.strip()
        ],
        "events_path": events_path,
    }


def _names(result: dict[str, Any]) -> list[str]:
    return [event["event"] for event in result["events"]]


def _transitions(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in result["events"] if event["event"] == "transition"]


def _assert_log_is_closed(result: dict[str, Any]) -> None:
    """Every stage_start paired, every boundary recorded, ending on trial_end.

    This is what "the bookkeeping survived the failure" means concretely, and
    it is asserted on EVERY failure case rather than in one place: the three
    fixes differ in where the exception comes from, but they all claim the same
    outcome for the log.
    """
    names = _names(result)
    assert names[-1] == "trial_end", names
    depth = 0
    for name in names:
        depth += 1 if name == "stage_start" else (-1 if name == "stage_end" else 0)
        assert depth in (0, 1), names
    assert depth == 0, (
        f"stage_start 없이 끝난 단계가 있다 / unpaired stage_start: {names}"
    )
    # One boundary per stage that started, no more and no fewer: the boundary
    # line is the only carrier of the stop_base outcome, so a missing one is a
    # base stop nobody can audit and a duplicate is a second stop_base_s sample.
    assert len(_transitions(result)) == names.count("stage_start"), names


def test_a_dead_camera_falls_back_to_the_direct_base_command(tmp_path: Path) -> None:
    """The blocker: the primary stop_base needs the very cameras that just died.

    build_hold_action calls get_observation, which on MobileAIRobot reads three
    RealSense cameras, so the mechanism that stops the base used to fail in
    exactly the case it exists for -- measured leaving the base at 0.4 m/s
    through the encoder flush. Here the camera dies at the start of stage 2:
    record_loop raises, the error handler's stop_base hits the same dead camera,
    and only the fallback can zero the base.
    """
    faults = _Faults()

    def kill_the_camera(call_number: int, captured: dict[str, Any]) -> None:
        if call_number == 2:
            captured["robot"]._base_velocity = dict(DRIVING_BASE)
            faults.observation_fails = True

    result = _run_chain(tmp_path, faults=faults, before_stage=kill_the_camera)

    assert isinstance(result["raised"], RuntimeError), result["raised"]
    assert result["base_at_encoder_flush"] == STOPPED_BASE, (
        "카메라가 죽어도 베이스는 서 있어야 한다 / the base is still driving into "
        f"the encoder flush: {result['base_at_encoder_flush']}"
    )
    assert result["robot"].base.commands == [(0.0, 0.0)], result["robot"].base.commands
    _assert_log_is_closed(result)
    paths = [event["stop_base_path"] for event in _transitions(result)]
    assert paths == ["primary", "fallback"], paths
    fallback = _transitions(result)[-1]
    assert fallback["reason"] == "stop_base_fallback"
    # No hold action was built, and the field says so -- but the base IS
    # stopped. A reader that took stop_action's absence for a failed stop would
    # report this working safety mechanism as a hazard, which is why
    # aggregate.stop_base_path keys off stop_base_path instead.
    assert fallback["stop_action"] is None
    assert "RealSense" in fallback["stop_base_error"]
    assert result["events"][-1]["reason"] == "exception"
    assert result["events"][-1]["episode_saved"] is False


def test_an_executor_error_stops_the_base_and_closes_the_log(tmp_path: Path) -> None:
    """The cameras are alive here, so the primary path is what must fire."""

    def blow_up(call_number: int, captured: dict[str, Any]) -> None:
        if call_number == 2:
            captured["robot"]._base_velocity = dict(DRIVING_BASE)
            raise RuntimeError("policy blew up mid-stage (injected)")

    result = _run_chain(tmp_path, before_stage=blow_up)

    assert isinstance(result["raised"], RuntimeError), result["raised"]
    assert result["base_at_encoder_flush"] == STOPPED_BASE
    _assert_log_is_closed(result)
    assert [event["stop_base_path"] for event in _transitions(result)] == [
        "primary",
        "primary",
    ]
    stage_ends = [event for event in result["events"] if event["event"] == "stage_end"]
    assert stage_ends[-1]["terminator"] == "error"
    assert result["events"][-1]["reason"] == "exception"
    assert result["events"][-1]["completed"] is False


def test_a_keyboard_interrupt_still_stops_the_base(tmp_path: Path) -> None:
    """KeyboardInterrupt is not an Exception, and that is the whole point.

    An ``except Exception`` in run_trial would drop the operator's reflex stop
    straight through, leaving the base driving through the unwind and the trial
    unresolved in the log -- while every other test stayed green. This case is
    that tidy-up's tripwire.
    """

    def interrupt(call_number: int, captured: dict[str, Any]) -> None:
        if call_number == 2:
            captured["robot"]._base_velocity = dict(DRIVING_BASE)
            raise KeyboardInterrupt

    result = _run_chain(tmp_path, before_stage=interrupt)

    assert isinstance(result["raised"], KeyboardInterrupt), result["raised"]
    assert result["base_at_encoder_flush"] == STOPPED_BASE
    _assert_log_is_closed(result)
    assert result["events"][-1]["reason"] == "exception"


def test_a_failing_stage_end_emit_does_not_outrun_the_base_stop(
    tmp_path: Path,
) -> None:
    """HARDWARE FIRST, BOOKKEEPING AFTER, asserted at the instant it matters.

    The measured case is OSError(28, 'No space left on device') out of the
    stage_end emit -- the classic failure on a robot PC writing video from three
    cameras. When bookkeeping ran first, that exception escaped with the base
    still holding the last policy velocity. The assertion is not "the base ends
    up at zero" (the error handler would do that anyway); it is that the base is
    ALREADY at zero at the moment the emit raises, which is only true if
    stop_base runs before it.
    """
    base_when_the_disk_filled: dict[str, Any] = {}
    stage_ends = {"n": 0}
    original_emit = event_stream.EventLog.emit
    holder: dict[str, Any] = {}

    def full_disk_emit(self: Any, event: str, **kwargs: Any) -> Any:
        if event == event_stream.EVENT_STAGE_END:
            stage_ends["n"] += 1
            if stage_ends["n"] == 2:
                base_when_the_disk_filled.update(holder["robot"]._base_velocity)
                raise OSError(28, "No space left on device")
        return original_emit(self, event, **kwargs)

    def leave_the_base_driving(call_number: int, captured: dict[str, Any]) -> None:
        holder["robot"] = captured["robot"]
        if call_number == 2:
            captured["robot"]._base_velocity = dict(DRIVING_BASE)

    result = _run_chain(
        tmp_path,
        after_stage=leave_the_base_driving,
        emit_patch=mock.patch.object(event_stream.EventLog, "emit", full_disk_emit),
    )

    assert isinstance(result["raised"], OSError), result["raised"]
    assert base_when_the_disk_filled == STOPPED_BASE, (
        "stage_end 기록이 베이스 정지보다 먼저 돌았다 / the base was still at "
        f"{base_when_the_disk_filled} when the stage_end emit ran, so bookkeeping "
        "overtook the hardware again"
    )
    assert result["base_at_encoder_flush"] == STOPPED_BASE
    # The emit that failed was the second stage_end, so that line is missing and
    # the handler writes its own error stage_end in its place -- the log is
    # still complete and still paired.
    _assert_log_is_closed(result)
    assert result["events"][-1]["reason"] == "exception"


def test_an_abort_before_the_first_frame_is_recorded_not_silent(
    tmp_path: Path,
) -> None:
    """Esc pressed before stage 1 starts: no frames, and no episode to save.

    The boundary line is emitted here too even though the chain crossed
    nothing -- an abort is exactly when the operator has decided something is
    wrong, so it is the last boundary whose base stop should go unrecorded.
    """
    result = _run_chain(tmp_path, preset_events={"stop_recording": True})

    assert result["raised"] is None, result["raised"]
    assert result["exit_code"] == cli.EXIT_ABORTED
    assert _names(result) == [
        "trial_start",
        "stage_start",
        "stage_end",
        "transition",
        "trial_end",
    ]
    _assert_log_is_closed(result)
    boundary = _transitions(result)[0]
    assert boundary["to_stage_id"] is None, "중단 경계는 다음 단계를 가리키지 않는다"
    assert boundary["stop_base_path"] == "primary"
    stage_end = [event for event in result["events"] if event["event"] == "stage_end"][
        0
    ]
    assert stage_end["terminator"] == "stop_recording"
    assert stage_end["frames"] == 0
    assert stage_end["elapsed_s"] == 0.0, (
        "record_loop에 들어가지도 않은 단계는 0.0이어야 한다 / aggregate uses "
        "elapsed_s > 0.0 to tell a stage that never ran from a slow one"
    )
    assert result["events"][-1]["reason"] == "aborted_empty"
    assert result["events"][-1]["episode_saved"] is False


def test_both_stop_paths_failing_is_visible_in_the_log(tmp_path: Path) -> None:
    """The base really is still moving here. The requirement is that it SHOWS.

    This is the case the review measured as byte-identical to a clean run: the
    final stop_base failed, save_episode ran with the base at 0.4 m/s, and
    events.jsonl said reason="completed" with exit code 0. The base cannot be
    saved when the observation is dead AND the base refuses the direct command;
    what can be saved is the operator's ability to find out.
    """
    faults = _Faults()

    def kill_everything_after_the_last_stage(
        call_number: int, captured: dict[str, Any]
    ) -> None:
        if call_number == 2:
            captured["robot"]._base_velocity = dict(DRIVING_BASE)
            faults.observation_fails = True

    result = _run_chain(
        tmp_path,
        faults=faults,
        base_accepts=False,
        after_stage=kill_everything_after_the_last_stage,
    )

    assert result["raised"] is None, result["raised"]
    # NOT EXIT_OK. This assertion is the round-3 blocker: the exit code used to
    # be derived from terminated_by alone, so the one outcome that means "the
    # base may still be driving" could not reach it and this run exited 0. A
    # batch script loops on the exit status and never opens events.jsonl, so
    # exit 0 here meant the next trial started with the base still moving.
    assert result["exit_code"] == cli.EXIT_BASE_STOP_FAILED, result["exit_code"]
    assert result["base_at_encoder_flush"] == DRIVING_BASE, (
        "이 케이스는 베이스가 계속 도는 상황을 재현해야 한다 / the case did not "
        "reproduce a moving base, so it proves nothing"
    )
    _assert_log_is_closed(result)
    final = _transitions(result)[-1]
    assert final["stop_base_path"] == "failed"
    assert final["reason"] == "stop_base_failed"
    assert final["to_stage_id"] is None
    assert final["stop_base_error"] is not None
    assert "set_cmd_vel" in final["stop_base_error"], final["stop_base_error"]
    # The run still "completed" -- every stage ran -- so inside the JSONL this
    # line is the ONLY thing separating it from a clean run. Asserted directly,
    # because its absence is what the review measured. The trial_end vocabulary
    # is deliberately NOT widened to say "the base failed": trial_end answers
    # what happened to the TRIAL, the transition answers what happened to the
    # hardware, and the exit code above is what carries the second one out of
    # the file.
    assert result["events"][-1]["reason"] == "completed"


def test_a_ctrl_c_inside_stop_base_is_labelled_fallback_not_failed(
    tmp_path: Path,
) -> None:
    """The indicator must not cry wolf on the most ordinary abort there is.

    The operator hits Ctrl+C while the boundary's get_observation is running.
    stop_base catches it, the fallback zeroes the base through set_cmd_vel, and
    the base IS stopped -- so the honest label is "fallback". It used to read
    "failed", the label that means "the base may still be moving": stop_base
    re-raised the interrupt and the outcome died with it, so the handler one
    frame up filed every escaping interrupt as a failed stop.

    That matters more now than it did before, because "failed" carries an exit
    code since this round. A routine Ctrl+C reported as a hazard is how the
    hazard indicator stops being read at all -- and the assertion pairs the label
    with the measured base velocity, so a label that says "failed" while the base
    reads 0.0 fails here rather than in someone's judgement on the day.
    """
    faults = _Faults()

    def interrupt_the_boundary(call_number: int, captured: dict[str, Any]) -> None:
        if call_number == 2:
            captured["robot"]._base_velocity = dict(DRIVING_BASE)
            faults.observation_interrupts = True

    result = _run_chain(tmp_path, faults=faults, after_stage=interrupt_the_boundary)

    assert isinstance(result["raised"], KeyboardInterrupt), result["raised"]
    # The fallback ran and it worked: THIS is what makes "failed" a lie.
    assert result["base_at_encoder_flush"] == STOPPED_BASE, (
        "Ctrl+C 경계에서 폴백이 베이스를 못 세웠다 / the fallback did not stop the "
        f"base: {result['base_at_encoder_flush']}"
    )
    assert result["robot"].base.commands[-1] == (0.0, 0.0)
    _assert_log_is_closed(result)
    final = _transitions(result)[-1]
    assert final["stop_base_path"] == "fallback", final
    assert final["reason"] == "stop_base_fallback", final
    # The primary error is still named, so the log says WHY the fallback ran.
    assert "KeyboardInterrupt" in final["stop_base_error"], final["stop_base_error"]
    assert result["events"][-1]["reason"] == "exception"


class _StubStage:
    """The two attributes load_bundles reads off a stage.

    Duck-typed rather than a real StageConfig because the wiring under test is
    "does the warm-up result reach the bundle", and a real StageConfig would drag
    a robot block and a terminator mapping into a test about one boolean.
    """

    def __init__(self, stage_id: str, policy_path: str) -> None:
        self.id = stage_id
        self.policy_path = policy_path


def test_load_bundles_records_whether_the_warm_up_ran() -> None:
    """The warm-up result must survive into the bundle and into trial_start.

    load_bundles used to call warm_up_bundle and DROP its return, so whether the
    pass ran reached the console and nowhere else -- and init_logging runs with
    log_file=None, so it was gone as soon as the terminal scrolled. Whether the
    warm-up ran is what decides if stage 0's `hertz` is biased low by
    first-inference CUDA/cuDNN init, and that hertz is the number the first-run
    procedure reads before the timeouts are fixed. Both directions are asserted:
    a test that only checks the True case passes on `warmed_up=True` hardcoded.
    """
    stages = [_StubStage("s4", "checkpoints/act_task04")]

    def _load(stage_id: str, policy_path: str, policy_config: Any, meta: Any) -> Any:
        return policies.PolicyBundle(
            stage_id=stage_id,
            policy_path=policy_path,
            config=_StubConfig(None),
            policy=_StubPolicy(),
            preprocessor=None,
            postprocessor=None,
        )

    for ran in (True, False):
        with (
            mock.patch.object(policies, "load_bundle", _load),
            # ran bound as a default: a bare closure over the loop variable
            # would make both iterations read the LAST value, so the False case
            # would silently re-test the True one.
            mock.patch.object(policies, "warm_up_bundle", lambda bundle, ran=ran: ran),
        ):
            bundles = policies.load_bundles(stages, {"s4": None}, None)
        assert bundles["s4"].warmed_up is ran
        assert policies.bundle_descriptor(bundles["s4"])["warmed_up"] is ran

    # A bundle nobody warmed reports false by default, which is what makes the
    # mock path (make_mock_bundle, deliberately never warmed) truthful without
    # mock_policy.py having to know this field exists.
    assert (
        policies.PolicyBundle(
            stage_id="s4",
            policy_path="mock://s4",
            config=_StubConfig(None),
            policy=_StubPolicy(),
            preprocessor=None,
            postprocessor=None,
        ).warmed_up
        is False
    )


class _StubFeature:
    """The one attribute warm_up_bundle reads off a PolicyFeature."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _StubConfig:
    def __init__(self, features: dict[str, _StubFeature] | None) -> None:
        self.input_features = features
        self.device = "cpu"


class _StubPolicy:
    def __init__(self, explode: bool = False) -> None:
        self.explode = explode
        self.batches: list[dict[str, Any]] = []
        self.resets = 0

    def select_action(self, batch: dict[str, Any]) -> Any:
        self.batches.append(batch)
        if self.explode:
            raise RuntimeError(
                "this policy wants a key input_features does not declare"
            )
        return None

    def reset(self) -> None:
        self.resets += 1


def _stub_bundle(policy: _StubPolicy, features: dict[str, _StubFeature] | None) -> Any:
    return policies.PolicyBundle(
        stage_id="s4",
        policy_path="stub",
        config=_StubConfig(features),
        policy=policy,
        preprocessor=None,
        postprocessor=None,
    )


def test_the_policy_warm_up_never_costs_a_run() -> None:
    """The warm-up is an optimisation, so its failure must cost a log line only.

    It runs before robot.connect() on EVERY real run, which makes it the one
    piece of this package that can kill a trial before the robot is even
    energised. The three cases are the three ways it can go: it works, the
    policy refuses the batch shaped from input_features (a language-conditioned
    checkpoint wanting `task`), or there are no input_features to shape one
    from. None of them may raise, and all of them must leave the action queue
    reset -- a queue left holding actions inferred from ZEROS would be popped by
    the first real frames of the stage.
    """
    policy = _StubPolicy()
    features = {
        "observation.state": _StubFeature((16,)),
        "observation.images.top": _StubFeature((3, 480, 640)),
    }
    assert policies.warm_up_bundle(_stub_bundle(policy, features)) is True
    assert policy.resets == 1
    shapes = {key: tuple(value.shape) for key, value in policy.batches[0].items()}
    # The batch dim is added and the declared shape is copied verbatim: a
    # warm-up at the wrong shape compiles the wrong kernels and moves nothing.
    assert shapes == {
        "observation.state": (1, 16),
        "observation.images.top": (1, 3, 480, 640),
    }, shapes

    refusing = _StubPolicy(explode=True)
    assert policies.warm_up_bundle(_stub_bundle(refusing, features)) is False
    assert refusing.resets == 1, "실패해도 큐는 비워야 한다 / reset must still run"

    undeclared = _StubPolicy()
    assert policies.warm_up_bundle(_stub_bundle(undeclared, None)) is False
    assert undeclared.batches == []


def test_the_writer_and_the_reader_share_one_stop_base_vocabulary() -> None:
    """transitions writes these strings; aggregate reads them. They must match.

    aggregate.py duplicates the vocabulary rather than importing it, on purpose:
    it is stdlib-only so an old log can be re-aggregated without lerobot or
    torch, and importing transitions would drag in lerobot.robots. Duplication
    is the price, and this assertion is what keeps the two copies honest --
    a third path added on the writer's side with no reader entry would be filed
    as a FAILED base stop, and a renamed reason would be too.
    """
    assert transitions.REASON_BY_STOP_PATH == aggregate.STOP_BASE_REASON_BY_PATH
    assert (
        transitions.STOP_PATH_PRIMARY,
        transitions.STOP_PATH_FALLBACK,
        transitions.STOP_PATH_FAILED,
    ) == (
        aggregate.STOP_PATH_PRIMARY,
        aggregate.STOP_PATH_FALLBACK,
        aggregate.STOP_PATH_FAILED,
    )


if __name__ == "__main__":
    # No pytest in this venv and it must not be mutated, so the file is its own
    # runner. Each case gets its own temporary root: they all write a dataset
    # under the same repo_id and LeRobotDataset.create is exist_ok=False.
    TESTS = [
        test_smoke_chain_runs_without_hardware,
        test_a_dead_camera_falls_back_to_the_direct_base_command,
        test_an_executor_error_stops_the_base_and_closes_the_log,
        test_a_keyboard_interrupt_still_stops_the_base,
        test_a_failing_stage_end_emit_does_not_outrun_the_base_stop,
        test_an_abort_before_the_first_frame_is_recorded_not_silent,
        test_both_stop_paths_failing_is_visible_in_the_log,
        test_a_ctrl_c_inside_stop_base_is_labelled_fallback_not_failed,
    ]
    for case in TESTS:
        with tempfile.TemporaryDirectory() as directory:
            case(Path(directory))
        print(f"[ok] {case.__name__}")
    test_the_policy_warm_up_never_costs_a_run()
    print("[ok] test_the_policy_warm_up_never_costs_a_run")
    test_load_bundles_records_whether_the_warm_up_ran()
    print("[ok] test_load_bundles_records_whether_the_warm_up_ran")
    test_the_writer_and_the_reader_share_one_stop_base_vocabulary()
    print(
        f"[ok] {test_the_writer_and_the_reader_share_one_stop_base_vocabulary.__name__}"
    )
    print(f"[smoke] {len(TESTS) + 3} passed")
