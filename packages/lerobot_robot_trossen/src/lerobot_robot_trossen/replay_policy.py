"""Replay a recorded episode's actions through the eval loop (opt in, bench only).

Why this exists
---------------
The latency switches in ``chunk_execution_patch.py`` are meant to make the base
turn the way the demonstrations did. A trained policy mixes that question with
its own errors (a different plan from a new start pose, drift after the first
turn). Replaying a demonstration's own ``action`` rows through the same eval
loop removes the policy from the comparison: whatever difference is left between
the demo's heading and the measured heading comes from the loop -- its rate, the
chunk-boundary stall, the command-to-velocity lag -- which is what the switches
target.

The stall has to be realistic for that to work, so by default the checkpoint
given with ``--policy.path`` still runs its real inference on every chunk
request and the result is thrown away. A ``sleep`` would not do: it releases the
GIL, so a background prefetch would look perfect in the bench while real
inference competes for the GIL with the base driver on the robot.

Usage
-----
Off by default. The eval command stays as it is, plus::

    LEROBOT_REPLAY_DATASET=kiroaiseoul/task10_move_to_beaker_shelf \\
    LEROBOT_REPLAY_EPISODES=46,72,35,34 \\
    uv run lerobot-record ... --dataset.repo_id=<user>/eval_replay_task10_... \\
        --policy.path=<the checkpoint whose inference cost to reproduce>

* Episode ``i`` of the run replays ``LEROBOT_REPLAY_EPISODES[i % n]``; a
  re-recorded episode replays the same one again.
* ``LEROBOT_REPLAY_RUN_POLICY=0`` skips the real inference (no stall at all).
* Refused unless ``--dataset.repo_id`` contains ``replay``: the variables are
  easy to leave exported in a shell, and a policy eval that silently turns into
  a replay would look like a very good policy.
* After the demo's last row the arms hold their last target and the base is
  commanded to stop.

The replayed ``action`` column is the demo's measured base velocity (kinesthetic
recording), so the bench compares the base's measured heading with a signal
that is itself one command-lag late -- read results as relative between switch
settings, not as an absolute error.
"""

import logging
import os

logger = logging.getLogger(__name__)

DATASET_ENVIRONMENT_VARIABLE = "LEROBOT_REPLAY_DATASET"
EPISODES_ENVIRONMENT_VARIABLE = "LEROBOT_REPLAY_EPISODES"
RUN_POLICY_ENVIRONMENT_VARIABLE = "LEROBOT_REPLAY_RUN_POLICY"

# The eval dataset name must carry this, so a leftover export cannot turn a
# policy eval into a replay unnoticed.
REQUIRED_REPOSITORY_MARKER = "replay"


def _parse_episodes(value: str) -> tuple[tuple[int, ...], str | None]:
    value = value.strip()
    if not value:
        return (), None
    try:
        episodes = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError:
        return (
            (),
            f"{EPISODES_ENVIRONMENT_VARIABLE}={value!r} is not a list of integers",
        )
    if any(episode < 0 for episode in episodes):
        return (), f"{EPISODES_ENVIRONMENT_VARIABLE}={value!r} has a negative index"
    return episodes, None


REPLAY_DATASET = os.getenv(DATASET_ENVIRONMENT_VARIABLE, "").strip() or None
REPLAY_EPISODES, EPISODES_PROBLEM = _parse_episodes(
    os.getenv(EPISODES_ENVIRONMENT_VARIABLE, "")
)
RUN_POLICY = os.getenv(RUN_POLICY_ENVIRONMENT_VARIABLE, "").strip().lower() not in (
    "0",
    "false",
    "no",
)


class IdentityPostprocessor:
    """Stands in for the policy postprocessor: replayed rows are already in robot units."""

    def __call__(self, action):
        return action.to("cpu")

    def reset(self) -> None:
        pass


class ReplayConfig:
    """The few config fields the eval loop and the chunk executor read."""

    def __init__(self, wrapped_config, extra_rows: int):
        self.n_action_steps = wrapped_config.n_action_steps
        # Rows returned per request: enough past the executed part that a base
        # lead of `extra_rows` ticks reads real demo rows instead of holding the
        # last one, so a replayed lead is a clean shift of the whole signal.
        self.chunk_size = wrapped_config.n_action_steps + extra_rows
        self.device = getattr(wrapped_config, "device", "cpu")
        self.use_amp = getattr(wrapped_config, "use_amp", False)
        self.temporal_ensemble_coeff = None
        self.compile_model = False
        self.type = f"replay({getattr(wrapped_config, 'type', 'policy')})"


class ReplayPolicy:
    """Returns demo action rows by observation tick; optionally runs real inference.

    Always driven through ``ChunkExecutor``, which passes the tick at which the
    observation was taken, so row ``t`` is sent at tick ``t`` whatever the
    prefetch setting.
    """

    def __init__(
        self,
        wrapped_policy,
        episode_actions: dict,
        episodes: tuple[int, ...],
        extra_rows: int,
        base_indices: tuple[int, ...],
        run_policy: bool,
    ):
        self.wrapped_policy = wrapped_policy
        self.config = ReplayConfig(wrapped_policy.config, extra_rows)
        self.episode_actions = episode_actions
        self.episodes = episodes
        self.base_indices = base_indices
        self.run_policy = run_policy
        self.current_episode: int | None = None
        self.current_actions = None

    def select_episode(self, recorded_episode_count: int) -> int:
        self.current_episode = self.episodes[
            recorded_episode_count % len(self.episodes)
        ]
        self.current_actions = self.episode_actions[self.current_episode]
        return self.current_episode

    def reset(self) -> None:
        # Tick counting lives in the executor; nothing to rewind here.
        pass

    def select_action(self, batch):  # pragma: no cover - the executor replaces it
        raise RuntimeError("ReplayPolicy must be driven through ChunkExecutor.")

    def predict_action_chunk(self, batch, observation_tick: int = 0):
        import torch

        if self.run_policy:
            # Reproduce the real inference cost (GPU time and GIL contention),
            # then discard the result.
            self.wrapped_policy.predict_action_chunk(batch)

        actions = self.current_actions
        row_count = actions.shape[0]
        wanted_rows = self.config.chunk_size
        first_row = min(observation_tick, row_count)
        rows = actions[first_row : first_row + wanted_rows]
        missing_rows = wanted_rows - rows.shape[0]
        if missing_rows > 0:
            # Past the end of the demo: hold the arms' last target, stop the base.
            final_row = actions[row_count - 1].clone()
            for base_index in self.base_indices:
                final_row[base_index] = 0.0
            rows = torch.cat([rows, final_row.expand(missing_rows, -1)], dim=0)
        device = batch_device(batch)
        return rows.to(device=device, dtype=torch.float32).unsqueeze(0)


def batch_device(batch):
    import torch

    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def load_episode_actions(
    repository_id: str, episodes: tuple[int, ...], root: str | None = None
):
    """Load the ``action`` rows of the given episodes, without videos.

    Returns ``(episode -> float32 tensor [frames, action_dim], action names)``.
    If ``episodes`` is empty every episode of the dataset is loaded, in order.
    ``root`` is only for tests; runs use the usual lerobot cache location.
    """
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repository_id,
        root=root,
        episodes=list(episodes) if episodes else None,
        download_videos=False,
    )
    action_names = list(dataset.features["action"]["names"])
    columns = dataset.hf_dataset.select_columns(["action", "episode_index"])
    # Slice the whole table: in the pinned `datasets` (4.x) indexing a column by
    # name returns a lazy Column object, not a tensor, while a slice returns
    # one stacked tensor per column.
    table = columns.with_format("torch")[:]
    all_actions = table["action"]
    episode_indices = table["episode_index"]
    if isinstance(all_actions, list):
        all_actions = torch.stack(all_actions)
    episode_actions = {}
    for episode in sorted({int(index) for index in episode_indices.tolist()}):
        mask = episode_indices == episode
        episode_actions[episode] = all_actions[mask].to(torch.float32)
    return episode_actions, action_names
