"""The record_loop wrapper: replay wiring, restoring on failure, banner, CSV.

These cover what sits around ``ChunkExecutor``: what the wrapper hands to the
real loop, what it restores when setting up fails, and what an operator reads
back afterwards.
"""

import logging
import time
from types import SimpleNamespace

import pytest
import torch
from fake_policies import (
    ACTION_NAMES,
    BASE_INDEX_BY_NAME,
    FakeChunkPolicy,
    FakeRobot,
    IdentityPostprocessor,
    decode,
    tick_batch,
)
from test_chunk_execution import attach, run_ticks, settings

from lerobot_robot_trossen import chunk_execution_patch, replay_policy
from lerobot_robot_trossen.chunk_execution_patch import (
    ExecutionLog,
    InferenceStuckError,
    ReplayPreparationError,
)


@pytest.fixture(autouse=True)
def clean_module_state(monkeypatch):
    monkeypatch.setattr(chunk_execution_patch, "_execution_log", None)
    yield
    chunk_execution_patch.join_unfinished_inferences()


def eval_dataset(repository_id="user/eval_task10", names=ACTION_NAMES):
    return SimpleNamespace(
        repo_id=repository_id,
        num_episodes=0,
        fps=21,
        features={"action": {"names": list(names)}},
    )


# ----- stuck inference -------------------------------------------------------------


def test_stuck_inference_stops_the_base_and_reraises():
    def stuck_loop(**arguments):
        raise InferenceStuckError("stuck")

    robot = FakeRobot()
    wrapped = chunk_execution_patch._wrap_record_loop(stuck_loop)
    with pytest.raises(InferenceStuckError):
        wrapped(robot=robot, policy=None, fps=21)
    assert robot.base_stops == 1


def test_a_loop_end_does_not_wait_for_the_prefetch_but_the_next_phase_does():
    policy = FakeChunkPolicy(action_steps=6, delay_seconds=0.3)
    executor = attach(policy, prefetch_ticks=2)
    run_ticks(policy, 5)  # a prefetch is now running
    start = time.perf_counter()
    executor.end_loop()
    assert time.perf_counter() - start < 0.1
    assert chunk_execution_patch._unfinished_inferences
    chunk_execution_patch.join_unfinished_inferences()
    assert policy.active_inferences == 0
    assert not chunk_execution_patch._unfinished_inferences


def test_a_hung_inference_from_an_earlier_loop_blocks_the_next_one(monkeypatch):
    policy = FakeChunkPolicy(action_steps=6, slow_calls=(1,))
    executor = attach(policy, prefetch_ticks=2)
    run_ticks(policy, 5)
    executor.end_loop()
    with pytest.raises(InferenceStuckError, match="earlier loop"):
        chunk_execution_patch.join_unfinished_inferences(timeout=0.05)
    chunk_execution_patch.join_unfinished_inferences()


def test_later_inferences_time_out_relative_to_the_slowest_so_far(monkeypatch):
    monkeypatch.setattr(chunk_execution_patch, "MINIMUM_INFERENCE_TIMEOUT_SECONDS", 0.1)
    policy = FakeChunkPolicy(action_steps=6, slow_calls=(2,))
    executor = attach(policy, prefetch_ticks=2)
    with pytest.raises(InferenceStuckError):
        run_ticks(policy, 20)
    executor.end_loop()


# ----- wrapper: replay wiring and restoring on failure --------------------------------


def demo_rows(frames=40):
    rows = torch.arange(frames).view(-1, 1) * 10 + torch.arange(4).view(1, -1)
    return rows.to(torch.float32)


@pytest.fixture
def replay_environment(monkeypatch):
    repository_id = "local/demo"
    monkeypatch.setattr(replay_policy, "REPLAY_DATASET", repository_id)
    monkeypatch.setattr(replay_policy, "REPLAY_EPISODES", (0,))
    monkeypatch.setattr(replay_policy, "EPISODES_PROBLEM", None)
    monkeypatch.setattr(replay_policy, "RUN_POLICY", False)
    monkeypatch.setattr(chunk_execution_patch, "SETTINGS", settings(base_lead_ticks=2))
    monkeypatch.setitem(
        chunk_execution_patch._replay_actions_cache,
        repository_id,
        ({0: demo_rows()}, list(ACTION_NAMES)),
    )
    return eval_dataset("user/eval_replay_task10")


def test_replay_swaps_policy_and_postprocessor_and_carries_the_lead(replay_environment):
    policy = FakeChunkPolicy(action_steps=10)
    postprocessor = object()
    arguments = {
        "policy": policy,
        "postprocessor": postprocessor,
        "dataset": replay_environment,
        "fps": 21,
    }
    state = chunk_execution_patch._prepare_loop((), arguments, [])
    replay = arguments["policy"]
    assert isinstance(replay, replay_policy.ReplayPolicy)
    assert isinstance(arguments["postprocessor"], replay_policy.IdentityPostprocessor)
    assert replay.config.chunk_size == 12  # n_action_steps + lead
    assert replay.base_indices == (2, 3)
    replay.reset()
    for tick in range(25):
        action = replay.select_action(tick_batch(tick))
        assert int(action[0, 0]) == tick * 10
        assert int(action[0, 3]) == (tick + 2) * 10 + 3
    chunk_execution_patch._finish_loop(state)


def test_replay_that_cannot_be_driven_stops_and_restores(
    replay_environment, monkeypatch
):
    monkeypatch.setattr(
        chunk_execution_patch,
        "attach_chunk_executor",
        lambda policy, settings=None: (None, "nope"),
    )
    policy, postprocessor = FakeChunkPolicy(action_steps=10), object()
    called = []
    wrapped = chunk_execution_patch._wrap_record_loop(
        lambda **arguments: called.append(arguments)
    )
    with pytest.raises(ReplayPreparationError, match="nope"):
        wrapped(
            policy=policy,
            postprocessor=postprocessor,
            dataset=replay_environment,
            fps=21,
        )
    assert called == []


def test_setup_failure_restores_the_arguments_and_the_policy(monkeypatch):
    monkeypatch.setattr(chunk_execution_patch, "SETTINGS", settings(prefetch_ticks=2))

    def broken_log():
        raise PermissionError("cannot write the log")

    monkeypatch.setattr(chunk_execution_patch, "_get_execution_log", broken_log)
    policy, postprocessor = FakeChunkPolicy(action_steps=10), object()
    received = []
    wrapped = chunk_execution_patch._wrap_record_loop(
        lambda **arguments: received.append(arguments)
    )
    wrapped(policy=policy, postprocessor=postprocessor, dataset=eval_dataset(), fps=21)
    assert received[0]["policy"] is policy
    assert received[0]["postprocessor"] is postprocessor
    assert "select_action" not in policy.__dict__
    assert "_chunk_executor" not in policy.__dict__


def test_unsupported_policy_types_are_left_alone():
    diffusion = FakeChunkPolicy()
    diffusion.config.type = "diffusion"
    executor, note = chunk_execution_patch.attach_chunk_executor(
        diffusion, settings(prefetch_ticks=2)
    )
    assert executor is None and "diffusion" in note


# ----- banner, CSV, summary -----------------------------------------------------------


def test_banner_is_a_warning_listing_every_set_switch(monkeypatch, caplog):
    values = {
        "LEROBOT_CHUNK_PREFETCH_TICKS": "5",
        "LEROBOT_CHUNK_PREFETCH_INLINE": "1",
        "LEROBOT_BASE_LEAD_TICKS": "2",
        "LEROBOT_REPLAY_DATASET": "local/demo",
        "LEROBOT_REPLAY_EPISODES": "0",
        "LEROBOT_REPLAY_RUN_POLICY": "0",
        "LEROBOT_CHUNK_EXECUTION_LOG": "/dev/null",
    }
    for variable, value in values.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setattr(chunk_execution_patch, "SETTINGS", settings())
    with caplog.at_level(logging.WARNING):
        chunk_execution_patch._prepare_loop(
            (),
            {"policy": FakeChunkPolicy(), "dataset": eval_dataset(), "fps": 21},
            [],
        )
    banners = [
        record for record in caplog.records if "Base latency switches" in record.message
    ]
    assert banners and banners[0].levelno == logging.WARNING
    for variable, value in values.items():
        assert f"{variable}={value}" in banners[0].message


def test_csv_header_is_pinned():
    assert ExecutionLog.COLUMNS == (
        "loop",
        "episode",
        "tick",
        "t_mono",
        "t_wall",
        "in_flight",
        "launched",
        "swapped",
        "blocked_ms",
        "inference_ms",
        "elapsed_ticks",
        "seam_theta_step",
        "lead_source",
        "planned_x_vel",
        "planned_theta_vel",
        "sent_x_vel",
        "sent_theta_vel",
        "arm_seam_raw",
        "arm_seam_joint",
        "arm_seam_sent",
        "blend_ticks",
        "prefix_gap",
        "arm_smooth_max",
    )


def test_csv_planned_command_and_seam_after_prefetch_swaps(tmp_path):
    log_path = tmp_path / "execution.csv"
    execution_log = ExecutionLog(str(log_path))
    policy = FakeChunkPolicy(action_steps=10)
    executor = attach(policy, execution_log, prefetch_ticks=3)
    run_ticks(policy, 40)
    executor.end_loop()
    lines = [
        line for line in log_path.read_text().splitlines() if not line.startswith("#")
    ]
    header, rows = lines[0].split(","), [line.split(",") for line in lines[1:]]
    column = {name: index for index, name in enumerate(header)}
    for tick, row in enumerate(rows):
        observation_tick, step, channel = decode(
            float(row[column["planned_theta_vel"]])
        )
        assert (observation_tick + step, channel) == (tick, 3)
    seams = [
        float(row[column["seam_theta_step"]])
        for row in rows
        if row[column["swapped"]] == "1" and row[column["seam_theta_step"]]
    ]
    # Old chunk's last step 9 from observation o, new chunk's step 3 from o + 7:
    # |(o + 7) * 1000 + 33 - (o * 1000 + 93)| = 6940.
    assert seams and all(seam == 6940.0 for seam in seams)


def test_in_flight_is_cleared_once_the_inference_finishes(tmp_path):
    log_path = tmp_path / "execution.csv"
    policy = FakeChunkPolicy(action_steps=10, delay_seconds=0.02)
    executor = attach(policy, ExecutionLog(str(log_path)), prefetch_ticks=4)
    for tick in range(12):
        policy.select_action(tick_batch(tick))
        time.sleep(0.015)
    executor.end_loop()
    lines = [
        line for line in log_path.read_text().splitlines() if not line.startswith("#")
    ]
    header, rows = lines[0].split(","), [line.split(",") for line in lines[1:]]
    column = {name: index for index, name in enumerate(header)}
    in_flight = [row[column["in_flight"]] for row in rows]
    # Launched at tick 6 with a 20 ms inference; over by tick 9, before the swap.
    assert in_flight[6] == "1"
    assert in_flight[9] == "0"


@pytest.mark.parametrize(
    ("inline", "thread_name"), [(False, "chunk-prefetch"), (True, "MainThread")]
)
def test_first_chunk_runs_where_the_prefetches_run(inline, thread_name):
    import threading

    names = []
    policy = FakeChunkPolicy(action_steps=10)
    original = policy.predict_action_chunk

    def recording(batch):
        names.append(threading.current_thread().name)
        return original(batch)

    policy.predict_action_chunk = recording
    attach(policy, prefetch_ticks=3, prefetch_inline=inline)
    policy.select_action(tick_batch(0))
    assert names == [thread_name]


def test_summary_suggests_a_prefetch_that_hides_the_inference(caplog):
    policy = FakeChunkPolicy(action_steps=10, delay_seconds=0.05)
    executor = attach(policy, prefetch_ticks=2)
    for tick in range(35):
        policy.select_action(tick_batch(tick))
        time.sleep(0.02)
    with caplog.at_level(logging.INFO):
        executor.end_loop()
    summary = next(
        record.message
        for record in caplog.records
        if "Chunk execution summary" in record.message
    )
    # ~50 ms of inference over ~20 ms ticks: at least ceil(50 / 20) + 1 = 4.
    suggested = int(summary.split("LEROBOT_CHUNK_PREFETCH_TICKS >= ")[1].split()[0])
    assert suggested >= 4


def test_every_set_variable_installs_the_patch(monkeypatch):
    for variable in chunk_execution_patch.SWITCH_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    assert not chunk_execution_patch.any_switch_requested()
    monkeypatch.setenv("LEROBOT_CHUNK_PREFETCH_INLINE", "1")
    assert chunk_execution_patch.any_switch_requested()


def test_identity_postprocessor_moves_to_cpu():
    assert IdentityPostprocessor()(torch.ones(1, 4)).device.type == "cpu"
    assert BASE_INDEX_BY_NAME == {"x.vel": 2, "theta.vel": 3}
