"""Replay bench: rows by observation tick, the executor on top, and its refusals."""

from types import SimpleNamespace

import pytest
import torch
from fake_policies import ACTION_NAMES, BASE_INDEX_BY_NAME, FakeChunkPolicy, tick_batch

from lerobot_robot_trossen import chunk_execution_patch, replay_policy
from lerobot_robot_trossen.chunk_execution_patch import (
    ChunkExecutionSettings,
    attach_chunk_executor,
)
from lerobot_robot_trossen.replay_policy import IdentityPostprocessor, ReplayPolicy

FRAMES = 30


def demo_actions(episode: int, frames: int = FRAMES) -> torch.Tensor:
    """Row t, channel c = episode * 10000 + t * 10 + c (easy to read back)."""
    rows = torch.arange(frames).view(-1, 1) * 10 + torch.arange(4).view(1, -1)
    return (episode * 10000 + rows).to(torch.float32)


def replay_for(wrapped, extra_rows=0, run_policy=False, episodes=(46, 72)):
    replay = ReplayPolicy(
        wrapped,
        {episode: demo_actions(episode) for episode in episodes},
        episodes,
        extra_rows=extra_rows,
        base_indices=(2, 3),
        run_policy=run_policy,
    )
    replay.select_episode(0)
    return replay


def row_and_channel(value: float) -> tuple[int, int]:
    value = int(round(float(value))) % 10000
    return value // 10, value % 10


def test_rows_follow_the_observation_tick_and_stop_the_base_at_the_end():
    replay = replay_for(FakeChunkPolicy(action_steps=10), extra_rows=2)
    chunk = replay.predict_action_chunk(tick_batch(0), observation_tick=25)
    assert chunk.shape == (1, 12, 4)
    assert row_and_channel(chunk[0, 0, 0]) == (25, 0)
    assert row_and_channel(chunk[0, 4, 1]) == (29, 1)
    past_end = chunk[0, 5]  # demo has rows 0..29
    assert torch.equal(past_end[:2], demo_actions(46)[29, :2])
    assert past_end[2].item() == 0.0 and past_end[3].item() == 0.0


def test_real_inference_runs_and_is_discarded():
    wrapped = FakeChunkPolicy(action_steps=10)
    replay = replay_for(wrapped, run_policy=True)
    chunk = replay.predict_action_chunk(tick_batch(7), observation_tick=3)
    assert wrapped.observation_ticks == [7]
    assert row_and_channel(chunk[0, 0, 0]) == (3, 0)


def test_episodes_cycle_by_recorded_count():
    replay = replay_for(FakeChunkPolicy(), episodes=(46, 72))
    assert [replay.select_episode(count) for count in range(4)] == [46, 72, 46, 72]


@pytest.mark.parametrize("prefetch_ticks", [0, 3])
def test_through_the_executor_row_t_is_sent_at_tick_t(prefetch_ticks):
    wrapped = FakeChunkPolicy(action_steps=10)
    replay = replay_for(wrapped, extra_rows=2)
    executor, note = attach_chunk_executor(
        replay,
        ChunkExecutionSettings(
            prefetch_ticks=prefetch_ticks,
            prefetch_inline=False,
            base_lead_ticks=2,
            execution_log_path=None,
        ),
    )
    assert executor is not None, note
    executor.begin_loop(BASE_INDEX_BY_NAME, IdentityPostprocessor(), None, 0)
    replay.reset()
    for tick in range(26):
        action = replay.select_action(tick_batch(tick))
        assert row_and_channel(action[0, 0]) == (tick, 0)
        # The lead is a clean shift of the whole demo: no held tail.
        assert row_and_channel(action[0, 3]) == (tick + 2, 3)
    assert executor._held_ticks == 0


def fake_eval_dataset(repository_id, names=ACTION_NAMES):
    return SimpleNamespace(
        repo_id=repository_id,
        num_episodes=1,
        features={"action": {"names": list(names)}},
    )


@pytest.fixture
def replay_source(monkeypatch):
    repository_id = "local/demo"
    monkeypatch.setattr(replay_policy, "REPLAY_DATASET", repository_id)
    monkeypatch.setattr(replay_policy, "REPLAY_EPISODES", (46, 72))
    monkeypatch.setattr(replay_policy, "EPISODES_PROBLEM", None)
    monkeypatch.setitem(
        chunk_execution_patch._replay_actions_cache,
        repository_id,
        ({46: demo_actions(46), 72: demo_actions(72)}, list(ACTION_NAMES)),
    )
    return repository_id


def test_replay_needs_replay_in_the_eval_repository_name(replay_source):
    replay, note = chunk_execution_patch._prepare_replay(
        FakeChunkPolicy(), fake_eval_dataset("user/eval_task10"), BASE_INDEX_BY_NAME
    )
    assert replay is None and "does not contain 'replay'" in note


def test_replay_stops_the_run_on_mismatched_action_names(replay_source):
    names = list(reversed(ACTION_NAMES))
    with pytest.raises(chunk_execution_patch.ReplayPreparationError, match="differ"):
        chunk_execution_patch._prepare_replay(
            FakeChunkPolicy(),
            fake_eval_dataset("user/eval_replay_task10", names),
            BASE_INDEX_BY_NAME,
        )


def test_replay_stops_the_run_on_missing_episodes(replay_source, monkeypatch):
    monkeypatch.setattr(replay_policy, "REPLAY_EPISODES", (46, 99))
    with pytest.raises(chunk_execution_patch.ReplayPreparationError, match="99"):
        chunk_execution_patch._prepare_replay(
            FakeChunkPolicy(),
            fake_eval_dataset("user/eval_replay_task10"),
            BASE_INDEX_BY_NAME,
        )


def test_replay_stops_the_run_when_the_demo_cannot_load(monkeypatch):
    monkeypatch.setattr(replay_policy, "REPLAY_DATASET", "local/does_not_exist")
    monkeypatch.setattr(replay_policy, "REPLAY_EPISODES", (0,))
    monkeypatch.setattr(replay_policy, "EPISODES_PROBLEM", None)

    def failing_load(repository_id, episodes):
        raise FileNotFoundError(repository_id)

    monkeypatch.setattr(replay_policy, "load_episode_actions", failing_load)
    with pytest.raises(chunk_execution_patch.ReplayPreparationError, match="load"):
        chunk_execution_patch._prepare_replay(
            FakeChunkPolicy(),
            fake_eval_dataset("user/eval_replay_task10"),
            BASE_INDEX_BY_NAME,
        )


def test_replay_warns_when_the_episode_is_shorter_than_the_demo(replay_source):
    dataset = fake_eval_dataset("user/eval_replay_task10")
    dataset.fps = 10  # 30 demo frames = 3 s
    _, note = chunk_execution_patch._prepare_replay(
        FakeChunkPolicy(), dataset, BASE_INDEX_BY_NAME, control_time_seconds=2
    )
    assert "ends mid-demo" in note


def test_end_padding_leaves_the_demo_untouched():
    replay = replay_for(FakeChunkPolicy(action_steps=10))
    before = replay.current_actions.clone()
    replay.predict_action_chunk(tick_batch(0), observation_tick=25)
    assert torch.equal(replay.current_actions, before)


def test_run_policy_off_skips_the_real_inference():
    wrapped = FakeChunkPolicy(action_steps=10)
    replay_for(wrapped, run_policy=False).predict_action_chunk(tick_batch(0), 0)
    assert wrapped.observation_ticks == []


def test_loads_a_real_local_dataset(tmp_path):
    """The pinned `datasets` returns lazy Column objects for a column index."""
    import numpy

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "demo"
    features = {
        "action": {"dtype": "float32", "shape": (4,), "names": list(ACTION_NAMES)},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["s"]},
    }
    dataset = LeRobotDataset.create(
        "local/demo", fps=10, features=features, root=root, use_videos=False
    )
    for episode in range(3):
        for frame in range(5 + episode):
            dataset.add_frame(
                {
                    "action": numpy.array(
                        [episode * 100 + frame, 0, 0, 0], dtype=numpy.float32
                    ),
                    "observation.state": numpy.zeros(1, numpy.float32),
                    "task": "demo",
                }
            )
        dataset.save_episode()
    dataset.finalize()

    actions, names = replay_policy.load_episode_actions("local/demo", (), root=root)
    assert names == ACTION_NAMES
    assert sorted(actions) == [0, 1, 2]
    assert actions[2][:, 0].tolist() == [200.0 + frame for frame in range(7)]
    assert actions[1].dtype == torch.float32

    subset, _ = replay_policy.load_episode_actions("local/demo", (2, 0), root=root)
    assert sorted(subset) == [0, 2]
    assert subset[0][:, 0].tolist() == [float(frame) for frame in range(5)]


def test_replay_is_prepared_once_per_policy(replay_source):
    policy = FakeChunkPolicy()
    dataset = fake_eval_dataset("user/eval_replay_task10")
    first, note = chunk_execution_patch._prepare_replay(
        policy, dataset, BASE_INDEX_BY_NAME
    )
    assert first is not None and "episode 72" in note  # num_episodes=1 -> second
    second, _ = chunk_execution_patch._prepare_replay(
        policy, dataset, BASE_INDEX_BY_NAME
    )
    assert second is first
