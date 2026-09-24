"""Switches for the three latency causes of base over-rotation (all off by default).

Why this exists
---------------
The Mobile AI base takes velocity commands and holds each one until the next
``send_action()``, so anything that delays or stretches the command stream turns
into extra rotation. Three separate latencies do that during a policy eval, and
each has its own switch so they can be measured one at a time in the same
session with the same checkpoint. This module holds two of them:

1. **Chunk boundary stall** (``LEROBOT_CHUNK_PREFETCH_TICKS``). A chunked policy
   infers only when its action queue runs dry, so the loop stops for the whole
   inference once per chunk (pi0: ~236 ms every 50 ticks) while the base keeps
   its last velocity. With the switch set to ``k``, the next chunk is inferred in
   a background thread from the observation ``k`` ticks before the boundary; at
   the boundary its first ``k`` actions (whose time has passed) are dropped, so
   action ``j`` of every chunk still runs ``j`` ticks after its observation.
   ``LEROBOT_CHUNK_PREFETCH_INLINE=1`` runs the same schedule with the inference
   in the loop thread: identical actions, stall kept. Compare the two to
   separate the stall from the side effect that re-planning now happens every
   ``n_action_steps - k`` ticks instead of every ``n_action_steps``.
2. **Loop period** -- not here: ``LEROBOT_BASE_SERIAL_REARM`` (removes the base
   driver's 20 ms receive wait) paired with ``--dataset.fps=21``. The prefetch
   leans on it: the driver binding holds the GIL through every Modbus
   transaction, so a background inference barely runs while the loop waits in
   them.
3. **Command-to-velocity lag** (``LEROBOT_BASE_LEAD_TICKS``). The demos' base
   labels are *measured* velocities, which trail the commands that produced them
   by ~0.1-0.16 s. With the switch set to ``d``, tick ``t`` sends the arms'
   action ``t`` but the base's action ``t + d`` of the same chunk. At the end of
   a chunk, where ``t + d`` runs past the chunk, the base takes the value the
   prefetched next chunk planned for ``t + d`` if that chunk has already arrived,
   and holds the chunk's last value otherwise -- so the base command never sits
   still at a chunk end as long as the prefetch lands ``d`` ticks early
   (``LEROBOT_CHUNK_PREFETCH_TICKS`` >= ``d`` + inference ticks). The inline
   control has the next chunk from its launch tick on, so with ``k > d`` prefetch
   and control send the same actions whenever the background inference is on
   time; a late one shows as ``held`` tail ticks in the execution log. A held
   tail is never later than no lead at all -- it only gives up the lead's
   ``d``-tick head start at that chunk end. This is Mobile ALOHA's
   ``BASE_DELAY`` applied at inference time; Mobile ALOHA instead re-plans ``d``
   ticks earlier, which would change the re-planning period too.

The switches change nothing unless set: with none of the variables set this
module is not installed at all. When the chunk executor is installed with the
prefetch and the lead both off (e.g. for the execution log alone), it reproduces
``select_action`` exactly.

Judging the switches
--------------------
* The recorded ``action`` column holds what was sent. With the base lead on,
  its base channels are the shifted commands, so the lead's effect only shows in
  the *measured* base velocity, not in any integral of the recorded commands.
* ``LEROBOT_CHUNK_EXECUTION_LOG=<file.csv>`` writes one row per tick: whether an
  inference was running, how long the tick blocked on inference, the planned
  (unshifted) and sent base command, and the seam at each chunk swap.
* At the start of every policy phase a WARNING banner lists every switch that is
  set and what it resolved to (``grep "Base latency switches" <log>``).

Safety
------
* A background inference that raises is redone synchronously.
* One that does not finish in time (60 s for a policy's first inference, which
  pays the CUDA warm-up; afterwards 5x the slowest one so far, at least 2 s)
  stops the base at once and ends the loop with ``InferenceStuckError``.
* No two inferences run at once. An inference still running when its loop ends
  is not waited for there -- that would hold the base at its last velocity
  before the caller gets to stop it -- but it is joined before the next policy
  phase starts and on every ``policy.reset()``.
* A replay that was asked for but cannot be prepared stops the run instead of
  driving the robot with the policy into a dataset named as a replay.
"""

import functools
import logging
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from lerobot_robot_trossen import replay_policy

logger = logging.getLogger(__name__)

PREFETCH_TICKS_VARIABLE = "LEROBOT_CHUNK_PREFETCH_TICKS"
PREFETCH_INLINE_VARIABLE = "LEROBOT_CHUNK_PREFETCH_INLINE"
BASE_LEAD_TICKS_VARIABLE = "LEROBOT_BASE_LEAD_TICKS"
EXECUTION_LOG_VARIABLE = "LEROBOT_CHUNK_EXECUTION_LOG"

SWITCH_VARIABLES = (
    PREFETCH_TICKS_VARIABLE,
    PREFETCH_INLINE_VARIABLE,
    BASE_LEAD_TICKS_VARIABLE,
    replay_policy.DATASET_ENVIRONMENT_VARIABLE,
    replay_policy.EPISODES_ENVIRONMENT_VARIABLE,
    replay_policy.RUN_POLICY_ENVIRONMENT_VARIABLE,
    EXECUTION_LOG_VARIABLE,
)

BASE_ACTION_NAMES = ("x.vel", "theta.vel")

# Policies whose select_action is "queue of predict_action_chunk outputs" and
# nothing else, so the executor can take its place. Diffusion and VQ-BeT fill
# observation queues inside select_action that predict_action_chunk then reads;
# SmolVLA fills its queues inside predict_action_chunk itself.
SUPPORTED_POLICY_TYPES = ("act", "pi0", "pi05", "smolvla")

FIRST_INFERENCE_TIMEOUT_SECONDS = 60.0
MINIMUM_INFERENCE_TIMEOUT_SECONDS = 2.0
SLOWEST_INFERENCE_TIMEOUT_FACTOR = 5.0

# Attribute that marks our record_loop wrapper. Deliberately not `__wrapped__`:
# loop_rate_log already wraps record_loop with functools.wraps and treats any
# `__wrapped__` as "already patched", so sharing that guard would silently skip
# one of the two patches.
WRAPPER_MARKER = "_chunk_execution_wrapper"


def switch_on(value: str) -> bool:
    """The repo's env switch idiom: anything but empty, 0, false or no is on."""
    return value.strip().lower() not in ("", "0", "false", "no")


class InferenceStuckError(RuntimeError):
    """A chunk inference did not finish in time; the base has been stopped."""


class ReplayPreparationError(RuntimeError):
    """A replay was asked for and cannot be set up; the run must not go ahead."""


def _read_tick_count(variable: str, problems: list[str]) -> int:
    value = os.getenv(variable, "").strip()
    if not value:
        return 0
    try:
        count = int(value)
    except ValueError:
        problems.append(f"{variable}={value!r} is not an integer; switch left off")
        return 0
    if count < 0:
        problems.append(f"{variable}={value!r} is negative; switch left off")
        return 0
    return count


@dataclass(frozen=True)
class ChunkExecutionSettings:
    prefetch_ticks: int
    prefetch_inline: bool
    base_lead_ticks: int
    execution_log_path: str | None
    problems: tuple[str, ...] = field(default=())

    @property
    def wants_executor(self) -> bool:
        return (
            self.prefetch_ticks > 0
            or self.base_lead_ticks > 0
            or self.execution_log_path is not None
        )


def read_settings() -> ChunkExecutionSettings:
    problems: list[str] = []
    prefetch_ticks = _read_tick_count(PREFETCH_TICKS_VARIABLE, problems)
    prefetch_inline = switch_on(os.getenv(PREFETCH_INLINE_VARIABLE, ""))
    if prefetch_inline and prefetch_ticks == 0:
        problems.append(
            f"{PREFETCH_INLINE_VARIABLE} does nothing without {PREFETCH_TICKS_VARIABLE}"
        )
    base_lead_ticks = _read_tick_count(BASE_LEAD_TICKS_VARIABLE, problems)
    execution_log_path = os.getenv(EXECUTION_LOG_VARIABLE, "").strip() or None
    if replay_policy.EPISODES_PROBLEM:
        problems.append(replay_policy.EPISODES_PROBLEM)
    if replay_policy.REPLAY_DATASET is None and (
        os.getenv(replay_policy.EPISODES_ENVIRONMENT_VARIABLE, "").strip()
        or os.getenv(replay_policy.RUN_POLICY_ENVIRONMENT_VARIABLE, "").strip()
    ):
        problems.append(
            "replay episode/run-policy variables do nothing without "
            f"{replay_policy.DATASET_ENVIRONMENT_VARIABLE}"
        )
    return ChunkExecutionSettings(
        prefetch_ticks=prefetch_ticks,
        prefetch_inline=prefetch_inline,
        base_lead_ticks=base_lead_ticks,
        execution_log_path=execution_log_path,
        problems=tuple(problems),
    )


# Frozen at import, like the other patches: the switches describe one run.
SETTINGS = read_settings()


def any_switch_requested() -> bool:
    """True if any of the variables is set at all, even to an off or bad value.

    Installing on any set variable (rather than only on effective ones) is what
    lets the banner say that a variable did nothing, instead of the run quietly
    behaving like a baseline.
    """
    return any(os.environ.get(variable, "").strip() for variable in SWITCH_VARIABLES)


def shift_base_channels(chunk, lead_ticks: int, base_indices: tuple[int, ...]):
    """Advance the base channels of ``chunk`` by ``lead_ticks`` steps.

    ``chunk`` is ``[batch, steps, action_dim]``. Step ``i`` of the result carries
    the arms' step ``i`` and the base's step ``min(i + lead_ticks, steps - 1)``:
    the last ``lead_ticks`` steps hold the chunk's final base value. Returns
    ``chunk`` itself when there is nothing to shift.
    """
    if lead_ticks <= 0 or not base_indices:
        return chunk
    import torch

    step_count = chunk.shape[1]
    source_steps = torch.clamp(
        torch.arange(step_count, device=chunk.device) + lead_ticks, max=step_count - 1
    )
    shifted = chunk.clone()
    base = list(base_indices)
    shifted[:, :, base] = chunk[:, source_steps][:, :, base]
    return shifted


class _CompletedInference:
    """An inference already run in the loop thread (inline prefetch)."""

    def __init__(self, result, observation_tick: int, inference_seconds: float):
        self.result = result
        self.error = None
        self.observation_tick = observation_tick
        self.inference_seconds = inference_seconds

    def finished(self) -> bool:
        return True

    def wait(self, timeout: float) -> bool:
        return True

    def join(self, timeout: float | None = None) -> bool:
        return True


class _BackgroundInference:
    """One chunk inference on a worker thread and its own CUDA stream."""

    def __init__(self, predict, batch, observation_tick: int, use_amp: bool, stream):
        import torch

        self.result = None
        self.error: BaseException | None = None
        self.observation_tick = observation_tick
        self.inference_seconds: float | None = None
        self._finished = threading.Event()
        self._ready_event = None
        if stream is not None:
            # The batch was produced on the loop thread's current stream; the
            # worker's stream waits for exactly that point before reading it.
            self._ready_event = torch.cuda.current_stream(stream.device).record_event()
        # The worker keeps `batch` referenced until its stream has synchronized,
        # so the caching allocator cannot hand those blocks out mid-inference.
        self._thread = threading.Thread(
            target=self._run,
            args=(predict, batch, use_amp, stream),
            name="chunk-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _run(self, predict, batch, use_amp: bool, stream) -> None:
        import contextlib

        import torch

        start = time.perf_counter()
        try:
            # inference_mode and autocast are thread-local, so the worker opens the
            # same contexts predict_action opens around select_action.
            with torch.inference_mode():
                if stream is None:
                    self.result = predict(batch, self.observation_tick)
                else:
                    autocast = (
                        torch.autocast(device_type="cuda")
                        if use_amp
                        else contextlib.nullcontext()
                    )
                    with autocast, torch.cuda.stream(stream):
                        stream.wait_event(self._ready_event)
                        self.result = predict(batch, self.observation_tick)
                        stream.synchronize()
        except BaseException as error:  # surfaced to the loop thread at the swap
            self.error = error
        finally:
            self.inference_seconds = time.perf_counter() - start
            self._finished.set()

    def finished(self) -> bool:
        return self._finished.is_set()

    def wait(self, timeout: float) -> bool:
        return self._finished.wait(timeout)

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()


# Inferences whose loop ended while they were still running. Joined before the
# next policy phase and on every reset, so they never overlap a new inference.
_unfinished_inferences: list[_BackgroundInference] = []


def join_unfinished_inferences(timeout: float = FIRST_INFERENCE_TIMEOUT_SECONDS):
    """Wait for inferences left running by an earlier loop; raise if one hangs."""
    while _unfinished_inferences:
        inference = _unfinished_inferences[0]
        if not inference.join(timeout):
            raise InferenceStuckError(
                f"An inference from an earlier loop is still running after "
                f"{timeout:g} s; refusing to start another one next to it."
            )
        _unfinished_inferences.pop(0)


def _park(pending) -> None:
    if isinstance(pending, _BackgroundInference) and not pending.finished():
        _unfinished_inferences.append(pending)


class ExecutionLog:
    """Per-tick CSV (``LEROBOT_CHUNK_EXECUTION_LOG``); ``#`` lines are run context."""

    COLUMNS = (
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
    )

    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, "a", buffering=1)
        if is_new:
            self._file.write(",".join(self.COLUMNS) + "\n")
        else:
            logger.warning(
                f"{EXECUTION_LOG_VARIABLE}: appending to the existing {self.path}; "
                "rows of this run are told apart by the `# process` line and the "
                "`loop` column."
            )
        self.write_comment(
            f"process pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}"
        )

    def write_comment(self, text: str) -> None:
        self._file.write(f"# {text}\n")

    def write_row(self, values: dict) -> None:
        cells = []
        for column in self.COLUMNS:
            value = values.get(column)
            if value is None:
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        self._file.write(",".join(cells) + "\n")

    def flush(self) -> None:
        self._file.flush()


_execution_log: ExecutionLog | None = None


def _get_execution_log() -> ExecutionLog | None:
    global _execution_log
    if SETTINGS.execution_log_path is None:
        return None
    if _execution_log is None:
        _execution_log = ExecutionLog(SETTINGS.execution_log_path)
    return _execution_log


def _policy_device(policy):
    import torch

    try:
        device = torch.device(getattr(policy.config, "device", "cpu") or "cpu")
    except (RuntimeError, TypeError):
        return None
    if device.type == "cuda" and not torch.cuda.is_available():
        return None
    return device


class ChunkExecutor:
    """Replaces a policy's ``select_action``/``reset`` with a chunk schedule.

    A tick is one ``select_action`` call. Within a tick the order is: swap to the
    next chunk if the current one is used up, launch the next inference if this
    tick is ``k`` ticks before the end, then return this tick's action.
    """

    def __init__(
        self,
        policy,
        prefetch_ticks: int,
        prefetch_inline: bool,
        base_lead_ticks: int,
    ):
        self.policy = policy
        self.prefetch_ticks = prefetch_ticks
        self.prefetch_inline = prefetch_inline
        self.base_lead_ticks = base_lead_ticks
        self.action_steps = policy.config.n_action_steps
        self._original_reset = policy.reset
        self._passes_observation_tick = isinstance(policy, replay_policy.ReplayPolicy)
        self._use_amp = bool(getattr(policy.config, "use_amp", False))
        self._stream = None
        self._slowest_inference_seconds: float | None = None
        # Rebound on every record_loop call (see begin_loop).
        self.base_indices: tuple[int, ...] = ()
        self.base_index_by_name: dict[str, int] = {}
        self.postprocessor = None
        self.execution_log: ExecutionLog | None = None
        self.loop_label = None
        self.episode_label = None
        self._pending = None
        self._reset_state()

    # ----- description ---------------------------------------------------------

    def describe(self) -> str:
        parts = []
        if self.prefetch_ticks:
            where = (
                "loop thread (inline control)" if self.prefetch_inline else "background"
            )
            parts.append(f"prefetch {self.prefetch_ticks} ticks, {where}")
        if self.base_lead_ticks:
            if self.base_indices:
                parts.append(f"base lead {self.base_lead_ticks} ticks")
            else:
                parts.append("base lead OFF (no x.vel/theta.vel in the action names)")
        if not parts:
            parts.append("synchronous (same schedule as select_action)")
        return ", ".join(parts)

    # ----- loop lifecycle ------------------------------------------------------

    def begin_loop(
        self,
        base_index_by_name,
        postprocessor,
        execution_log,
        episode_label,
        loop_label=None,
    ):
        self.base_index_by_name = dict(base_index_by_name)
        self.base_indices = tuple(
            base_index_by_name[name]
            for name in BASE_ACTION_NAMES
            if name in base_index_by_name
        )
        self.postprocessor = postprocessor
        self.execution_log = execution_log
        self.episode_label = episode_label
        self.loop_label = loop_label

    def end_loop(self) -> None:
        """Summarize and flush. An inference still running is parked, not joined."""
        pending, self._pending = self._pending, None
        _park(pending)
        self._log_summary()
        if self.execution_log is not None:
            self.execution_log.flush()

    def _reset_state(self) -> None:
        self._tick = 0
        self._executing = None
        self._position = 0
        self._current_chunk = None
        self._current_observation_tick = 0
        self._current_elapsed = 0
        self._swap_records: list[dict] = []
        self._tick_times: list[float] = []
        self._held_ticks = 0
        self._next_filled_ticks = 0
        # Loop-thread time spent inferring since the last swap: the inline control
        # stalls at the launch tick rather than at the swap.
        self._blocked_seconds_since_swap = 0.0

    def reset_policy(self) -> None:
        """Stands in for ``policy.reset``: join in-flight work, then reset."""
        pending, self._pending = self._pending, None
        _park(pending)
        join_unfinished_inferences()
        self._original_reset()
        self._reset_state()

    # ----- inference -----------------------------------------------------------

    def _predict(self, batch, observation_tick: int):
        if self._passes_observation_tick:
            return self.policy.predict_action_chunk(
                batch, observation_tick=observation_tick
            )
        return self.policy.predict_action_chunk(batch)

    def _uses_worker(self) -> bool:
        return self.prefetch_ticks > 0 and not self.prefetch_inline

    def _worker_stream(self, batch):
        import torch

        device = _policy_device(self.policy) or replay_policy.batch_device(batch)
        if device.type != "cuda":
            return None
        if self._stream is None or self._stream.device != device:
            self._stream = torch.cuda.Stream(device=device)
        return self._stream

    def _start_worker(self, batch, observation_tick: int) -> _BackgroundInference:
        return _BackgroundInference(
            self._predict,
            batch,
            observation_tick,
            self._use_amp,
            self._worker_stream(batch),
        )

    def _timeout_seconds(self) -> float:
        if self._slowest_inference_seconds is None:
            return FIRST_INFERENCE_TIMEOUT_SECONDS
        return max(
            MINIMUM_INFERENCE_TIMEOUT_SECONDS,
            SLOWEST_INFERENCE_TIMEOUT_FACTOR * self._slowest_inference_seconds,
        )

    def _note_inference_seconds(self, seconds: float | None) -> None:
        if seconds is None:
            return
        if self._slowest_inference_seconds is None or seconds > (
            self._slowest_inference_seconds
        ):
            self._slowest_inference_seconds = seconds

    def _collect(self, pending):
        """Wait for ``pending``; return (result or None, seconds blocked).

        ``pending`` stays referenced by the caller until this returns, so a
        timeout leaves it where end_loop can park it and the next phase joins it.
        """
        timeout = self._timeout_seconds()
        start = time.perf_counter()
        finished = pending.wait(timeout)
        blocked_seconds = time.perf_counter() - start
        if not finished:
            raise InferenceStuckError(
                f"Chunk inference did not finish within {timeout:g} s."
            )
        if pending.error is not None:
            logger.warning(
                f"Background chunk inference failed ({pending.error!r}); "
                "inferring synchronously instead."
            )
            return None, blocked_seconds
        self._note_inference_seconds(pending.inference_seconds)
        result = pending.result
        if self._uses_worker() and getattr(result, "is_cuda", False):
            import torch

            # Allocated on the worker's stream, read from now on by the loop's.
            result.record_stream(torch.cuda.current_stream(result.device))
        return result, blocked_seconds

    def _infer_now(self, batch, tick: int):
        """Inference whose result the loop waits for right away."""
        if self._uses_worker():
            # Same stream and allocator pool as the prefetches, including the
            # first chunk, so the pool is warm before the first real prefetch.
            self._pending = self._start_worker(batch, tick)
            result, blocked_seconds = self._collect(self._pending)
            pending, self._pending = self._pending, None
            if result is None:
                raise pending.error
            return result, blocked_seconds, pending.inference_seconds
        start = time.perf_counter()
        result = self._predict(batch, tick)
        seconds = time.perf_counter() - start
        self._note_inference_seconds(seconds)
        return result, seconds, seconds

    # ----- schedule ------------------------------------------------------------

    def select_action(self, batch):
        tick = self._tick
        self._tick_times.append(time.perf_counter())
        tick_record = {
            "in_flight": 0,
            "launched": 0,
            "swapped": 0,
            "blocked_ms": 0.0,
        }
        if self._executing is None or self._position >= self._executing.shape[1]:
            self._swap(batch, tick, tick_record)
        if (
            self.prefetch_ticks
            and self._pending is None
            and self._position == self._executing.shape[1] - self.prefetch_ticks
        ):
            self._launch(batch, tick, tick_record)
        if self._pending is not None and not self._pending.finished():
            tick_record["in_flight"] = 1

        action = self._executing[:, self._position]
        chunk_step = self._current_elapsed + self._position
        action, lead_source = self._apply_lead_tail(action, tick, chunk_step)
        if self.execution_log is not None:
            self._write_tick(tick, tick_record, action, chunk_step, lead_source)
        self._position += 1
        self._tick += 1
        return action

    def _launch(self, batch, tick: int, tick_record: dict) -> None:
        tick_record["launched"] = 1
        if self.prefetch_inline:
            start = time.perf_counter()
            result = self._predict(batch, tick)
            seconds = time.perf_counter() - start
            tick_record["blocked_ms"] += seconds * 1e3
            self._blocked_seconds_since_swap += seconds
            self._pending = _CompletedInference(result, tick, seconds)
        else:
            self._pending = self._start_worker(batch, tick)

    def _swap(self, batch, tick: int, tick_record: dict) -> None:
        result = None
        observation_tick = tick
        blocked_seconds = 0.0
        inference_seconds = None
        pending = self._pending
        if pending is not None:
            result, blocked_seconds = self._collect(pending)
            self._pending = None
            if result is not None:
                observation_tick = pending.observation_tick
                inference_seconds = pending.inference_seconds
        if result is None:
            result, extra_blocked, inference_seconds = self._infer_now(batch, tick)
            blocked_seconds += extra_blocked
            observation_tick = tick

        elapsed = tick - observation_tick
        seam = self._seam_step(result, elapsed)
        self._install(result, observation_tick, elapsed)

        chunk_blocked_seconds = self._blocked_seconds_since_swap + blocked_seconds
        self._blocked_seconds_since_swap = 0.0
        tick_record["swapped"] = 1
        tick_record["blocked_ms"] += blocked_seconds * 1e3
        tick_record["inference_ms"] = (
            None if inference_seconds is None else inference_seconds * 1e3
        )
        tick_record["elapsed_ticks"] = elapsed
        tick_record["seam_theta_step"] = seam
        self._swap_records.append(
            {
                "tick": tick,
                "blocked_ms": chunk_blocked_seconds * 1e3,
                "inference_ms": None
                if inference_seconds is None
                else inference_seconds * 1e3,
                "seam_theta_step": seam,
            }
        )

    def _install(self, chunk, observation_tick: int, elapsed: int) -> None:
        shifted = shift_base_channels(chunk, self.base_lead_ticks, self.base_indices)
        executing = shifted[:, : self.action_steps]
        if elapsed >= executing.shape[1]:
            logger.warning(
                f"Chunk arrived {elapsed} ticks after its observation, past its "
                f"{executing.shape[1]} actions; running only its last action."
            )
            elapsed = executing.shape[1] - 1
        self._executing = executing[:, elapsed:]
        self._current_chunk = chunk
        self._current_observation_tick = observation_tick
        self._current_elapsed = elapsed
        self._position = 0

    def _apply_lead_tail(self, action, tick: int, chunk_step: int):
        """Return this tick's action and where its base command came from.

        In the last ``d`` steps of a chunk the lead reads past the chunk's end,
        where ``shift_base_channels`` holds the last value. If the prefetched next
        chunk has already arrived -- for the inline control, from its launch tick
        on -- its plan covers those ticks, so the base takes the value it planned
        for ``tick + d`` instead (``"next"``); otherwise it holds (``"held"``).
        Holding keeps the base's command unchanged for up to ``d`` ticks at that
        chunk end, giving up the lead's head start exactly where a stop command
        can land.
        """
        if not self.base_lead_ticks or not self.base_indices:
            return action, None
        if chunk_step + self.base_lead_ticks <= self._current_chunk.shape[1] - 1:
            return action, "chunk"
        pending = self._pending
        if (
            pending is not None
            and pending.finished()
            and pending.error is None
            and pending.result is not None
        ):
            next_chunk = pending.result
            next_step = tick - pending.observation_tick + self.base_lead_ticks
            if 0 <= next_step < next_chunk.shape[1]:
                base = list(self.base_indices)
                filled = action.clone()
                filled[:, base] = next_chunk[:, next_step][:, base].to(
                    device=filled.device, dtype=filled.dtype
                )
                self._next_filled_ticks += 1
                return filled, "next"
        self._held_ticks += 1
        return action, "held"

    # ----- diagnostics ---------------------------------------------------------

    def _unnormalized_base(self, action_row):
        """(x.vel, theta.vel) of one ``[batch, action_dim]`` row in robot units."""
        if self.postprocessor is None or not self.base_index_by_name:
            return None, None
        try:
            unnormalized = self.postprocessor(action_row.clone())
            values = []
            for name in BASE_ACTION_NAMES:
                index = self.base_index_by_name.get(name)
                values.append(None if index is None else float(unnormalized[0, index]))
            return tuple(values)
        except Exception:
            logger.debug("Could not unnormalize an action for the log", exc_info=True)
            return None, None

    def _seam_step(self, new_chunk, elapsed: int):
        """|first theta.vel of the new chunk - last one of the old|, before any lead."""
        if self._current_chunk is None or "theta.vel" not in self.base_index_by_name:
            return None
        last_step = self._current_elapsed + self._executing.shape[1] - 1
        _, old_theta = self._unnormalized_base(self._current_chunk[:, last_step])
        first_step = min(elapsed, new_chunk.shape[1] - 1)
        _, new_theta = self._unnormalized_base(new_chunk[:, first_step])
        if old_theta is None or new_theta is None:
            return None
        return abs(new_theta - old_theta)

    def _write_tick(self, tick, tick_record, action, chunk_step, lead_source) -> None:
        planned_x, planned_theta = self._unnormalized_base(
            self._current_chunk[:, min(chunk_step, self._current_chunk.shape[1] - 1)]
        )
        if self.base_lead_ticks and self.base_indices:
            sent_x, sent_theta = self._unnormalized_base(action)
        else:
            sent_x, sent_theta = planned_x, planned_theta
        self.execution_log.write_row(
            {
                "loop": self.loop_label,
                "episode": self.episode_label,
                "tick": tick,
                # perf_counter, like LEROBOT_BASE_VEL_LOG's t_mono, so the two CSVs join.
                "t_mono": time.perf_counter(),
                "t_wall": time.time(),
                **tick_record,
                "lead_source": lead_source,
                "planned_x_vel": planned_x,
                "planned_theta_vel": planned_theta,
                "sent_x_vel": sent_x,
                "sent_theta_vel": sent_theta,
            }
        )

    def _log_summary(self) -> None:
        records = self._swap_records
        if not records:
            return
        # The first swap of a loop is the unavoidable (and possibly cold) first
        # inference; the prefetch can only hide the ones after it.
        later = records[1:]
        blocked = [record["blocked_ms"] for record in later]
        inference = [
            record["inference_ms"]
            for record in later
            if record["inference_ms"] is not None
        ]
        seams = [
            record["seam_theta_step"]
            for record in later
            if record["seam_theta_step"] is not None
        ]
        intervals = [
            later_time - earlier_time
            for earlier_time, later_time in zip(self._tick_times, self._tick_times[1:])
        ]
        parts = [
            f"ticks={self._tick}",
            f"swaps={len(records)}",
            f"schedule: {self.describe()}",
        ]
        if blocked:
            parts.append(
                f"loop blocked per later chunk median={statistics.median(blocked):.0f} "
                f"ms max={max(blocked):.0f} ms"
            )
        if inference:
            inference_95th_percentile = _percentile(inference, 0.95)
            parts.append(
                f"inference median={statistics.median(inference):.0f} ms "
                f"p95={inference_95th_percentile:.0f} ms"
            )
            if intervals:
                tick_interval_ms = statistics.median(intervals) * 1e3
                suggested = math.ceil(inference_95th_percentile / tick_interval_ms) + 1
                what = "hides it"
                if self.base_lead_ticks and self.base_indices:
                    # The lead's tail needs the next chunk d ticks before the swap.
                    suggested += self.base_lead_ticks
                    what = (
                        f"hides it and fills the {self.base_lead_ticks}-tick lead tail"
                    )

                where = (
                    "measured while the loop ran"
                    if self._uses_worker()
                    else "measured with the loop stopped, so an underestimate "
                    "for a background prefetch"
                )
                parts.append(
                    f"tick interval median={tick_interval_ms:.1f} ms -> "
                    f"{PREFETCH_TICKS_VARIABLE} >= {suggested} {what} ({where})"
                )
        if seams:
            parts.append(
                f"seam |d theta.vel| median={statistics.median(seams):.3f} "
                f"max={max(seams):.3f} rad/s"
            )
        if self.base_lead_ticks and self.base_indices:
            parts.append(
                f"base lead tail ticks: from next chunk={self._next_filled_ticks}, "
                f"held={self._held_ticks}"
            )
        logger.info(
            f"Chunk execution summary (loop {self.loop_label}, episode "
            f"{self.episode_label}): " + "; ".join(parts)
        )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[position]


def _policy_type(policy) -> str:
    return str(getattr(getattr(policy, "config", None), "type", ""))


def attach_chunk_executor(policy, settings: ChunkExecutionSettings | None = None):
    """Install a ``ChunkExecutor`` on ``policy``; return ``(executor, note)``.

    ``executor`` is None when the policy cannot be driven chunk-wise; ``note``
    says why, or what was switched off. Attaching twice returns the first one.
    ``settings`` defaults to the module's frozen ``SETTINGS`` at call time.
    """
    if settings is None:
        settings = SETTINGS
    existing = policy.__dict__.get("_chunk_executor")
    if existing is not None:
        return existing, ""
    config = getattr(policy, "config", None)
    action_steps = getattr(config, "n_action_steps", None)
    policy_type = _policy_type(policy)
    if not hasattr(policy, "predict_action_chunk") or not action_steps:
        return None, "policy has no predict_action_chunk/n_action_steps"
    if policy_type not in SUPPORTED_POLICY_TYPES and not isinstance(
        policy, replay_policy.ReplayPolicy
    ):
        return None, (
            f"policy type {policy_type!r} is not one whose select_action is a plain "
            f"chunk queue ({', '.join(SUPPORTED_POLICY_TYPES)})"
        )
    if getattr(config, "temporal_ensemble_coeff", None) is not None:
        return (
            None,
            "temporal ensembling infers every tick; there is no chunk to schedule",
        )
    rtc_enabled = getattr(policy, "_rtc_enabled", None)
    if callable(rtc_enabled) and rtc_enabled():
        return None, "RTC is enabled in the policy config"

    notes = []
    prefetch_ticks = settings.prefetch_ticks
    if prefetch_ticks and getattr(config, "compile_model", False):
        notes.append("prefetch off: compiled models record per-thread CUDA graphs")
        prefetch_ticks = 0
    if prefetch_ticks and 2 * prefetch_ticks > action_steps:
        notes.append(
            f"prefetch off: {PREFETCH_TICKS_VARIABLE}={prefetch_ticks} needs "
            f"n_action_steps >= {2 * prefetch_ticks} (policy has {action_steps})"
        )
        prefetch_ticks = 0
    if prefetch_ticks and settings.base_lead_ticks >= prefetch_ticks:
        notes.append(
            f"{BASE_LEAD_TICKS_VARIABLE}={settings.base_lead_ticks} is not below "
            f"{PREFETCH_TICKS_VARIABLE}={prefetch_ticks}: the lead's chunk tail starts "
            "no later than the next chunk's launch, so tail ticks hold until an "
            "inference lands (with the inline control too, when the lead is longer)"
        )

    executor = ChunkExecutor(
        policy,
        prefetch_ticks=prefetch_ticks,
        prefetch_inline=settings.prefetch_inline,
        base_lead_ticks=settings.base_lead_ticks,
    )
    # Instance attributes shadow the class methods; the class is left untouched.
    policy.select_action = executor.select_action
    policy.reset = executor.reset_policy
    policy._chunk_executor = executor
    return executor, "; ".join(notes)


def detach_chunk_executor(policy) -> None:
    """Undo ``attach_chunk_executor`` (used when a loop cannot be set up)."""
    for name in ("select_action", "reset", "_chunk_executor"):
        policy.__dict__.pop(name, None)


# ----- record_loop wiring --------------------------------------------------------

_replay_actions_cache: dict = {}
_fork_revision: str | None = None
_loop_counter = 0


def _fork_git_revision() -> str:
    global _fork_revision
    if _fork_revision is None:
        try:
            _fork_revision = subprocess.run(
                ["git", "describe", "--always", "--dirty"],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            ).stdout.strip()
        except Exception:
            _fork_revision = "unknown"
    return _fork_revision


def _action_index_by_name(dataset) -> dict[str, int]:
    try:
        names = list(dataset.features["action"]["names"])
    except Exception:
        return {}
    return {name: names.index(name) for name in BASE_ACTION_NAMES if name in names}


def _prepare_replay(policy, dataset, base_index_by_name, control_time_seconds=None):
    """Return ``(replay policy or None, note)`` for this record_loop call.

    Returns None only for the one refusal that leaves a normal policy eval
    running: the eval dataset is not named as a replay (a variable left
    exported). Anything that goes wrong after that raises
    ``ReplayPreparationError``.
    """
    repository_id = replay_policy.REPLAY_DATASET
    eval_repository = (
        str(getattr(dataset, "repo_id", "")) if dataset is not None else ""
    )
    if replay_policy.REQUIRED_REPOSITORY_MARKER not in eval_repository:
        return None, (
            f"refused, running the policy: --dataset.repo_id={eval_repository!r} does "
            f"not contain {replay_policy.REQUIRED_REPOSITORY_MARKER!r}"
        )
    if replay_policy.EPISODES_PROBLEM:
        raise ReplayPreparationError(replay_policy.EPISODES_PROBLEM)
    try:
        if repository_id not in _replay_actions_cache:
            _replay_actions_cache[repository_id] = replay_policy.load_episode_actions(
                repository_id, replay_policy.REPLAY_EPISODES
            )
    except Exception as error:
        raise ReplayPreparationError(
            f"could not load {repository_id}: {error!r}"
        ) from error
    episode_actions, action_names = _replay_actions_cache[repository_id]
    eval_names = list(dataset.features["action"]["names"])
    if action_names != eval_names:
        raise ReplayPreparationError(
            f"{repository_id} action names {action_names} differ from this run's "
            f"{eval_names}"
        )
    episodes = replay_policy.REPLAY_EPISODES or tuple(sorted(episode_actions))
    missing = [episode for episode in episodes if episode not in episode_actions]
    if missing:
        raise ReplayPreparationError(f"episodes {missing} not found in {repository_id}")

    replay = policy.__dict__.get("_replay_policy")
    if replay is None:
        replay = replay_policy.ReplayPolicy(
            policy,
            episode_actions,
            episodes,
            extra_rows=SETTINGS.base_lead_ticks,
            base_indices=tuple(base_index_by_name.values()),
            run_policy=replay_policy.RUN_POLICY,
        )
        policy._replay_policy = replay
    replay.base_indices = tuple(base_index_by_name.values())
    episode = replay.select_episode(int(getattr(dataset, "num_episodes", 0)))
    inference = "real inference, discarded" if replay.run_policy else "no inference"
    note = f"episode {episode} of {repository_id} ({inference})"
    demo_frames = replay.current_actions.shape[0]
    fps = getattr(dataset, "fps", None)
    if control_time_seconds and fps and demo_frames / fps > control_time_seconds:
        note += (
            f"; WARNING: the demo has {demo_frames} frames = {demo_frames / fps:.1f} s "
            f"at {fps} fps but the episode is {control_time_seconds:g} s, so it ends "
            "mid-demo with the base still moving"
        )
    return replay, note


def _banner_switches() -> str:
    set_variables = [
        f"{variable}={os.environ[variable]}"
        for variable in SWITCH_VARIABLES
        if os.environ.get(variable, "").strip()
    ]
    return " ".join(set_variables) or "(none set)"


@dataclass
class _LoopState:
    executor: ChunkExecutor | None


def _stop_base(robot) -> None:
    base = getattr(robot, "base", None)
    set_command = getattr(base, "set_cmd_vel", None)
    if set_command is None:
        return
    try:
        if not set_command(0.0, 0.0):
            logger.warning(
                "Stopping the base after a stuck inference reported failure."
            )
    except Exception:
        logger.exception("Could not stop the base after a stuck inference.")


def _prepare_loop(args, kwargs, attached_policies: list) -> _LoopState | None:
    global _loop_counter
    policy = kwargs.get("policy")
    fps = kwargs.get("fps")
    dataset = kwargs.get("dataset")

    if policy is None:
        return None
    if args:
        logger.warning(
            "record_loop was called with positional arguments; the base latency "
            "switches only handle keyword calls and are skipped for this loop."
        )
        return None

    # Inferences left running by the previous loop finish before anything new.
    join_unfinished_inferences()
    _loop_counter += 1
    loop_label = _loop_counter
    execution_log = _get_execution_log()  # fails here, before anything is attached

    base_index_by_name = _action_index_by_name(dataset)
    executor_policy = policy
    replay_note = None
    if replay_policy.REPLAY_DATASET is not None:
        replay, replay_note = _prepare_replay(
            policy, dataset, base_index_by_name, kwargs.get("control_time_s")
        )
        if replay is not None:
            executor_policy = replay

    executor = None
    executor_note = "not needed"
    if SETTINGS.wants_executor or executor_policy is not policy:
        executor, executor_note = attach_chunk_executor(executor_policy)
        if executor is None:
            if executor_policy is not policy:
                raise ReplayPreparationError(
                    f"the replay cannot be driven: {executor_note}"
                )
            executor_note = f"not attached: {executor_note}"
        else:
            attached_policies.append(executor_policy)
            episode_label = getattr(dataset, "num_episodes", None)
            executor.begin_loop(
                base_index_by_name,
                replay_policy.IdentityPostprocessor()
                if executor_policy is not policy
                else kwargs.get("postprocessor"),
                execution_log,
                episode_label,
                loop_label,
            )
            executor_note = executor.describe() + (
                f" ({executor_note})" if executor_note else ""
            )

    parts = [
        f"Base latency switches: {_banner_switches()}",
        f"executor: {executor_note}",
    ]
    if replay_note is not None:
        parts.append(f"replay: {replay_note}")
    if SETTINGS.problems:
        parts.append("problems: " + "; ".join(SETTINGS.problems))
    parts.append(f"loop {loop_label}")
    parts.append(f"fork {_fork_git_revision()}")
    banner = " | ".join(parts)
    logger.warning(banner)
    if execution_log is not None:
        pretrained_path = getattr(policy.config, "pretrained_path", None)
        execution_log.write_comment(
            f"loop start loop={loop_label} episode={getattr(dataset, 'num_episodes', None)} "
            f"dataset={getattr(dataset, 'repo_id', None)} fps={fps} "
            f"checkpoint={pretrained_path} | {banner}"
        )

    # Only now touch the caller's arguments: everything above can still fail.
    if executor_policy is not policy:
        kwargs["policy"] = executor_policy
        kwargs["postprocessor"] = replay_policy.IdentityPostprocessor()
    return _LoopState(executor=executor)


def _finish_loop(state: _LoopState | None) -> None:
    if state is not None and state.executor is not None:
        state.executor.end_loop()


def _wrap_record_loop(original):
    """Wrap ``record_loop`` to attach the executor and scope the switches to it."""

    @functools.wraps(original)
    def record_loop(*args, **kwargs):
        state = None
        original_arguments = dict(kwargs)
        attached_policies: list = []
        try:
            state = _prepare_loop(args, kwargs, attached_policies)
        except Exception as error:
            kwargs.clear()
            kwargs.update(original_arguments)
            for policy in attached_policies:
                detach_chunk_executor(policy)
            if isinstance(error, (ReplayPreparationError, InferenceStuckError)):
                logger.error(f"Base latency switches: {error}")
                raise
            # Never let the switches break a recording otherwise.
            logger.exception(
                "Could not apply the base latency switches; this loop runs without them."
            )
            state = None
        try:
            return original(*args, **kwargs)
        except InferenceStuckError:
            _stop_base(kwargs.get("robot"))
            raise
        finally:
            try:
                _finish_loop(state)
            except Exception:
                logger.exception("Could not finish the base latency switches cleanly.")

    setattr(record_loop, WRAPPER_MARKER, True)
    return record_loop


def is_installed(function) -> bool:
    """True if ``function`` or anything it wraps is our record_loop wrapper."""
    seen = set()
    while function is not None and id(function) not in seen:
        seen.add(id(function))
        if getattr(function, WRAPPER_MARKER, False):
            return True
        function = getattr(function, "__wrapped__", None)
    return False


def apply_chunk_execution_patch() -> bool:
    """Install the record_loop wrapper if any switch is set; return True if installed.

    Never raises: lerobot imports this package automatically by name prefix and
    treats any exception as a failed plugin, which would take the
    ``mobileai_robot`` registration down with it.
    """
    if not any_switch_requested():
        return False
    try:
        from lerobot.scripts import lerobot_record

        original = getattr(lerobot_record, "record_loop", None)
        if original is None:
            logger.warning(
                "lerobot.scripts.lerobot_record.record_loop is missing; the base "
                "latency switches are not installed."
            )
            return False
        if is_installed(original):
            return True
        wrapped = _wrap_record_loop(original)
        lerobot_record.record_loop = wrapped
        for module in list(sys.modules.values()):
            if module is None or module is lerobot_record:
                continue
            try:
                if getattr(module, "record_loop", None) is original:
                    module.record_loop = wrapped
            except Exception:  # modules with exotic __getattr__
                continue
        return True
    except Exception:
        logger.exception(
            "Could not install the base latency switches; running without them."
        )
        return False
