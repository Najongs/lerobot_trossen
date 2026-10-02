"""Base lead chunk tail: when it fills from the next chunk, when it holds, what it logs.

The fill may only use a prefetched chunk that has finished without error -- on
CUDA the worker stores its result before its stream synchronizes, so "has a
result" alone is not "usable".
"""

import logging
from types import SimpleNamespace

import pytest
from fake_policies import (
    FakeChunkPolicy,
    tick_batch,
)
from test_chunk_execution import attach, decoded, run_ticks

from lerobot_robot_trossen.chunk_execution_patch import ExecutionLog


def test_base_lead_past_the_next_chunks_plan_holds_instead_of_indexing_out():
    # d=8 > k=4: tail starts at chunk step 2, before the launch at step 6; the
    # next chunk (observation 6) plans steps 0..9, so ticks 8,9 (step 10,11) hold.
    policy = FakeChunkPolicy(action_steps=10)
    executor = attach(policy, prefetch_ticks=4, prefetch_inline=True, base_lead_ticks=8)
    actions = run_ticks(policy, 10)
    base = [decoded(a, 3)[:2] for a in actions]
    assert base[6:8] == [(6, 8), (6, 9)]  # next chunk's plan for tick + 8
    assert all(b == (0, 9) for b in base[2:6] + base[8:10])  # held
    assert executor._next_filled_ticks == 2 and executor._held_ticks == 6


@pytest.mark.parametrize(
    "finished, error, has_result",
    [
        (False, None, True),
        (True, RuntimeError("sync failed"), True),
        (True, None, False),
    ],
    ids=["not-finished", "errored-after-result", "no-result"],
)
def test_base_lead_holds_when_the_pending_chunk_is_not_usable(
    finished, error, has_result
):
    policy = FakeChunkPolicy(action_steps=10)
    executor = attach(policy, base_lead_ticks=2)
    run_ticks(policy, 8)
    result = policy.predict_action_chunk(tick_batch(7)) if has_result else None
    executor._pending = SimpleNamespace(
        result=result, error=error, observation_tick=7, finished=lambda: finished
    )
    for tick in (8, 9):
        assert decoded(policy.select_action(tick_batch(tick)), 3)[:2] == (0, 9)
    executor._pending = None
    assert executor._held_ticks == 2 and executor._next_filled_ticks == 0


def test_execution_log_labels_each_tail_tick_and_summary_counts_them(tmp_path, caplog):
    log_path = tmp_path / "execution.csv"
    policy = FakeChunkPolicy(action_steps=10)
    executor = attach(
        policy,
        ExecutionLog(str(log_path)),
        prefetch_ticks=4,
        prefetch_inline=True,
        base_lead_ticks=2,
    )
    run_ticks(policy, 20)
    with caplog.at_level(logging.INFO):
        executor.end_loop()
    lines = [
        line for line in log_path.read_text().splitlines() if not line.startswith("#")
    ]
    column = lines[0].split(",").index("lead_source")
    sources = [line.split(",")[column] for line in lines[1:]]
    # position 0..5 chunk, 6..9 swap-shifted; with k=4 each chunk is 6 ticks long
    # after the first: chunk 0 = ticks 0..9 (tail 8,9), later chunks start at step 4.
    assert sources[8:10] == ["next", "next"] and "held" not in sources
    assert "from next chunk=" + str(sources.count("next")) + ", held=0" in caplog.text


def test_lead_source_is_empty_when_the_lead_is_off(tmp_path):
    log_path = tmp_path / "execution.csv"
    policy = FakeChunkPolicy(action_steps=6)
    executor = attach(policy, ExecutionLog(str(log_path)), prefetch_ticks=2)
    run_ticks(policy, 8)
    executor.end_loop()
    lines = [
        line for line in log_path.read_text().splitlines() if not line.startswith("#")
    ]
    column = lines[0].split(",").index("lead_source")
    assert {line.split(",")[column] for line in lines[1:]} == {""}
