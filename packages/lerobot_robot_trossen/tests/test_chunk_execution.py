"""Chunk executor: prefetch schedule, base lead, safety paths, equivalence.

Drives ``ChunkExecutor`` directly with policies whose actions encode which
observation and which chunk step produced them, so each test can assert the
schedule tick by tick. Whole-process behavior (what gets installed for which
environment) is in ``test_patch_install.py``.
"""

import time

import pytest
import torch
from fake_policies import (
    BASE_INDEX_BY_NAME,
    FakeChunkPolicy,
    IdentityPostprocessor,
    decode,
    tick_batch,
    tiny_act_policy,
)

from lerobot_robot_trossen import chunk_execution_patch
from lerobot_robot_trossen.chunk_execution_patch import (
    ChunkExecutionSettings,
    ExecutionLog,
    InferenceStuckError,
    attach_chunk_executor,
    shift_base_channels,
)


def settings(prefetch_ticks=0, prefetch_inline=False, base_lead_ticks=0):
    return ChunkExecutionSettings(
        prefetch_ticks=prefetch_ticks,
        prefetch_inline=prefetch_inline,
        base_lead_ticks=base_lead_ticks,
        execution_log_path=None,
    )


def attach(policy, execution_log=None, **setting_values):
    executor, note = attach_chunk_executor(policy, settings(**setting_values))
    assert executor is not None, note
    executor.begin_loop(BASE_INDEX_BY_NAME, IdentityPostprocessor(), execution_log, 0)
    policy.reset()
    return executor


def run_ticks(policy, tick_count):
    return [policy.select_action(tick_batch(tick)) for tick in range(tick_count)]


def run_paced_ticks(policy, tick_count, seconds=0.02):
    """Like run_ticks, but leaves a background inference time to land in between."""
    actions = []
    for tick in range(tick_count):
        actions.append(policy.select_action(tick_batch(tick)))
        time.sleep(seconds)
    return actions


def decoded(action, channel):
    return decode(action[0, channel])


# ----- base lead as a pure function ----------------------------------------------


def test_shift_moves_only_base_channels_and_holds_the_tail():
    chunk = torch.arange(6 * 4, dtype=torch.float32).view(1, 6, 4)
    shifted = shift_base_channels(chunk, 2, (2, 3))
    assert torch.equal(shifted[:, :, :2], chunk[:, :, :2])
    for step in range(6):
        source = min(step + 2, 5)
        assert torch.equal(shifted[0, step, 2:], chunk[0, source, 2:])


def test_shift_is_identity_without_lead_or_base():
    chunk = torch.randn(1, 5, 4)
    assert shift_base_channels(chunk, 0, (2, 3)) is chunk
    assert shift_base_channels(chunk, 3, ()) is chunk


# ----- equivalence with select_action ---------------------------------------------


def test_synchronous_executor_reproduces_act_select_action():
    reference = tiny_act_policy(seed=0)
    candidate = tiny_act_policy(seed=0)
    candidate.load_state_dict(reference.state_dict())
    executor, note = attach_chunk_executor(candidate, settings())
    assert executor is not None, note
    executor.begin_loop(BASE_INDEX_BY_NAME, None, None, 0)

    generator = torch.Generator().manual_seed(1)
    for _episode in range(2):
        reference.reset()
        candidate.reset()
        for _tick in range(25):
            batch = {
                "observation.state": torch.randn(1, 6, generator=generator),
                "observation.environment_state": torch.randn(1, 3, generator=generator),
            }
            with torch.inference_mode():
                expected = reference.select_action(dict(batch))
                actual = candidate.select_action(dict(batch))
            assert actual.dtype == expected.dtype
            assert torch.equal(actual, expected)


def test_synchronous_schedule_matches_the_queue():
    policy = FakeChunkPolicy(action_steps=10)
    attach(policy)
    actions = run_ticks(policy, 35)
    assert policy.observation_ticks == [0, 10, 20, 30]
    for tick, action in enumerate(actions):
        observation_tick, step, _ = decoded(action, 0)
        assert observation_tick == (tick // 10) * 10
        assert step == tick % 10


# ----- prefetch schedule ---------------------------------------------------------


@pytest.mark.parametrize("inline", [False, True])
def test_prefetch_keeps_every_action_on_its_own_tick(inline):
    policy = FakeChunkPolicy(action_steps=10)
    executor = attach(policy, prefetch_ticks=3, prefetch_inline=inline)
    actions = run_ticks(policy, 60)
    executor.end_loop()
    chunk_execution_patch.join_unfinished_inferences()  # the last prefetch
    # First chunk from tick 0, then one observation every n - k = 7 ticks.
    assert policy.observation_ticks == [0, 7, 14, 21, 28, 35, 42, 49, 56]
    for tick, action in enumerate(actions):
        for channel in range(4):
            observation_tick, step, decoded_channel = decoded(action, channel)
            assert decoded_channel == channel
            assert observation_tick + step == tick
            assert step < 10


def test_inline_control_sends_the_same_actions_as_background_prefetch():
    # Same actions as long as the background inference lands before the lead's
    # tail starts (k >= d + inference ticks), which pacing guarantees here.
    background = FakeChunkPolicy(action_steps=8)
    inline = FakeChunkPolicy(action_steps=8)
    background_executor = attach(background, prefetch_ticks=4, base_lead_ticks=2)
    inline_executor = attach(
        inline, prefetch_ticks=4, prefetch_inline=True, base_lead_ticks=2
    )
    expected_actions = run_paced_ticks(background, 40)
    actual_actions = run_paced_ticks(inline, 40)
    for expected, actual in zip(expected_actions, actual_actions):
        assert torch.equal(expected, actual)
    assert background_executor._held_ticks == inline_executor._held_ticks == 0


def test_prefetch_with_half_the_chunk_launches_right_after_the_swap():
    policy = FakeChunkPolicy(action_steps=6)
    executor = attach(policy, prefetch_ticks=3)
    actions = run_ticks(policy, 24)
    executor.end_loop()
    chunk_execution_patch.join_unfinished_inferences()  # the last prefetch
    assert policy.observation_ticks == [0, 3, 6, 9, 12, 15, 18, 21]
    for tick, action in enumerate(actions):
        observation_tick, step, _ = decoded(action, 0)
        assert observation_tick + step == tick


def test_slow_background_inference_blocks_but_never_overlaps():
    policy = FakeChunkPolicy(action_steps=6, delay_seconds=0.05)
    attach(policy, prefetch_ticks=2)
    actions = run_ticks(policy, 30)
    assert policy.most_concurrent_inferences == 1
    for tick, action in enumerate(actions):
        observation_tick, step, _ = decoded(action, 0)
        assert observation_tick + step == tick


# ----- base lead inside the schedule ------------------------------------------------


def test_base_lead_with_prefetch_never_holds_when_the_next_chunk_is_in():
    policy = FakeChunkPolicy(action_steps=10)
    executor = attach(policy, prefetch_ticks=4, base_lead_ticks=2)
    actions = run_paced_ticks(policy, 40)
    for tick, action in enumerate(actions):
        arm_observation, arm_step, _ = decoded(action, 0)
        base_observation, base_step, _ = decoded(action, 3)
        assert arm_observation + arm_step == tick
        # Every tick, the base carries some plan's command for tick + 2 --
        # the current chunk's, or at a chunk end the next chunk's.
        assert base_observation + base_step == tick + 2
    assert executor._held_ticks == 0
    assert executor._next_filled_ticks > 0


def test_base_lead_holds_the_tail_when_the_next_chunk_is_late():
    # Calls 1 and later are prefetches that take 1 s: none lands in the tail.
    policy = FakeChunkPolicy(action_steps=10)
    policy.slow_calls = (1, 2, 3)
    executor = attach(policy, prefetch_ticks=4, base_lead_ticks=2)
    actions = run_ticks(policy, 12)
    for tick, action in enumerate(actions[:10]):
        arm_observation, arm_step, _ = decoded(action, 0)
        base_observation, base_step, _ = decoded(action, 3)
        assert arm_observation + arm_step == tick
        assert base_observation == arm_observation
        assert base_step == min(arm_step + 2, 9)  # the last two ticks hold step 9
    assert executor._held_ticks == 2 and executor._next_filled_ticks == 0
    executor.end_loop()


def test_base_lead_without_prefetch_holds_the_tail():
    policy = FakeChunkPolicy(action_steps=10)
    executor = attach(policy, base_lead_ticks=3)
    for tick, action in enumerate(run_ticks(policy, 20)):
        _, arm_step, _ = decoded(action, 0)
        _, base_step, _ = decoded(action, 3)
        assert base_step == min(arm_step + 3, 9)
    assert executor._held_ticks == 6 and executor._next_filled_ticks == 0


def test_lead_not_below_prefetch_is_flagged():
    policy = FakeChunkPolicy(action_steps=10)
    executor, note = attach_chunk_executor(
        policy, settings(prefetch_ticks=3, base_lead_ticks=3)
    )
    assert executor.prefetch_ticks == 3
    assert "LEROBOT_BASE_LEAD_TICKS=3 is not below" in note


def test_summary_suggestion_adds_the_lead(caplog):
    policy = FakeChunkPolicy(action_steps=10, delay_seconds=0.05)
    executor = attach(policy, prefetch_ticks=4, base_lead_ticks=3)
    run_paced_ticks(policy, 35)
    with caplog.at_level("INFO"):
        executor.end_loop()
    summary = next(
        r.message for r in caplog.records if "Chunk execution summary" in r.message
    )
    suggested = int(summary.split("LEROBOT_CHUNK_PREFETCH_TICKS >= ")[1].split()[0])
    # ~50 ms of inference over ~20 ms ticks is ceil(50 / 20) + 1 = 4 without a lead.
    assert suggested >= 4 + 3
    assert "fills the 3-tick lead tail" in summary


def test_base_lead_holds_the_tail_when_the_prefetch_failed():
    policy = FakeChunkPolicy(action_steps=10, failing_calls=(1,))
    executor = attach(policy, prefetch_ticks=4, base_lead_ticks=2)
    actions = run_paced_ticks(policy, 10)
    for action in actions[8:]:
        _, base_step, _ = decoded(action, 3)
        assert base_step == 9
    assert executor._held_ticks == 2


def test_base_lead_reads_real_steps_when_the_chunk_is_longer():
    policy = FakeChunkPolicy(action_steps=10, chunk_size=12)
    executor = attach(policy, base_lead_ticks=2)
    for action in run_ticks(policy, 30):
        _, arm_step, _ = decoded(action, 0)
        _, base_step, _ = decoded(action, 2)
        assert base_step == arm_step + 2
    assert executor._held_ticks == 0


# ----- safety paths --------------------------------------------------------------


def test_failed_background_inference_falls_back_to_a_synchronous_one(caplog):
    # Call 0 is the first chunk; call 1 is the first prefetch, which fails.
    policy = FakeChunkPolicy(action_steps=6, failing_calls=(1,))
    attach(policy, prefetch_ticks=2)
    with caplog.at_level("WARNING"):
        actions = run_ticks(policy, 20)
    assert "inferring synchronously instead" in caplog.text
    for tick, action in enumerate(actions):
        observation_tick, step, _ = decoded(action, 0)
        assert observation_tick + step == tick


def test_stuck_background_inference_raises(monkeypatch):
    monkeypatch.setattr(chunk_execution_patch, "MINIMUM_INFERENCE_TIMEOUT_SECONDS", 0.2)
    policy = FakeChunkPolicy(action_steps=6, slow_calls=(1,))
    executor = attach(policy, prefetch_ticks=2)
    with pytest.raises(InferenceStuckError):
        run_ticks(policy, 10)
    executor.end_loop()
    chunk_execution_patch.join_unfinished_inferences()


def test_reset_drops_in_flight_work_before_the_next_inference():
    policy = FakeChunkPolicy(action_steps=6, delay_seconds=0.1)
    executor = attach(policy, prefetch_ticks=2)
    run_ticks(policy, 5)  # the prefetch for the second chunk is now in flight
    assert executor._pending is not None
    policy.reset()
    assert executor._pending is None
    actions = run_ticks(policy, 8)
    assert policy.most_concurrent_inferences == 1
    observation_tick, step, _ = decoded(actions[0], 0)
    assert (observation_tick, step) == (0, 0)


def test_the_next_reset_joins_work_left_running_by_another_policy():
    first = FakeChunkPolicy(action_steps=6, delay_seconds=0.1)
    second = FakeChunkPolicy(action_steps=6)
    first_executor = attach(first, prefetch_ticks=2)
    run_ticks(first, 5)
    deadline = time.monotonic() + 1.0
    while first.active_inferences == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert first.active_inferences == 1
    first_executor.end_loop()  # parks the running prefetch instead of waiting
    attach(second, prefetch_ticks=2)  # second.reset() joins it first
    assert first.active_inferences == 0
    run_ticks(second, 3)


# ----- what the executor refuses --------------------------------------------------


def test_refuses_temporal_ensembling_and_rtc():
    ensembling = FakeChunkPolicy()
    ensembling.config.temporal_ensemble_coeff = 0.01
    assert attach_chunk_executor(ensembling, settings(prefetch_ticks=2))[0] is None

    real_time_chunking = FakeChunkPolicy()
    real_time_chunking._rtc_enabled = lambda: True
    assert (
        attach_chunk_executor(real_time_chunking, settings(prefetch_ticks=2))[0] is None
    )

    class NoChunks:
        config = FakeChunkPolicy().config

        def reset(self):
            pass

    assert attach_chunk_executor(NoChunks(), settings(prefetch_ticks=2))[0] is None


def test_prefetch_is_switched_off_but_lead_kept_when_unsafe():
    compiled = FakeChunkPolicy(action_steps=10)
    compiled.config.compile_model = True
    executor, note = attach_chunk_executor(
        compiled, settings(prefetch_ticks=2, base_lead_ticks=1)
    )
    assert executor.prefetch_ticks == 0 and executor.base_lead_ticks == 1
    assert "compiled" in note

    short = FakeChunkPolicy(action_steps=6)
    executor, note = attach_chunk_executor(short, settings(prefetch_ticks=4))
    assert executor.prefetch_ticks == 0
    assert "n_action_steps >= 8" in note


def test_attaching_twice_keeps_the_first_executor():
    policy = FakeChunkPolicy()
    first, _ = attach_chunk_executor(policy, settings(prefetch_ticks=2))
    second, _ = attach_chunk_executor(policy, settings(base_lead_ticks=3))
    assert first is second


# ----- execution log --------------------------------------------------------------


def test_execution_log_writes_one_row_per_tick(tmp_path):
    log_path = tmp_path / "execution.csv"
    execution_log = ExecutionLog(str(log_path))
    # A 20 ms inference, so the in-flight flag is still set when the loop looks
    # even on a starved CPU (an instant fake can finish before the check).
    policy = FakeChunkPolicy(action_steps=6, delay_seconds=0.02)
    executor = attach(policy, execution_log, prefetch_ticks=2, base_lead_ticks=1)
    run_ticks(policy, 20)
    executor.end_loop()
    lines = [
        line for line in log_path.read_text().splitlines() if not line.startswith("#")
    ]
    header, rows = lines[0].split(","), [line.split(",") for line in lines[1:]]
    assert header == list(ExecutionLog.COLUMNS)
    assert len(rows) == 20
    column = {name: index for index, name in enumerate(header)}
    sources = {row[column["lead_source"]] for row in rows}
    assert "chunk" in sources and sources <= {"chunk", "next", "held"}
    swapped = [row for row in rows if row[column["swapped"]] == "1"]
    assert all(row[column["elapsed_ticks"]] in ("0", "2") for row in swapped)
    assert any(row[column["in_flight"]] == "1" for row in rows)
    # Planned is the unshifted step, sent the shifted one, both in robot units.
    first = rows[0]
    assert float(first[column["planned_theta_vel"]]) == 3.0  # tick 0, step 0, ch 3
    assert float(first[column["sent_theta_vel"]]) == 13.0  # step 1 after the lead
