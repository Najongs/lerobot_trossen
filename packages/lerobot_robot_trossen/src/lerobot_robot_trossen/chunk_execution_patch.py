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

Two more switches smooth the arms where a new chunk takes over. With the
prefetch, the new chunk comes from an observation ``k`` ticks old and from its
own noise sample, so its first executed action need not continue from where the
old chunk left the arms (9/25 pi0 eval: 12% of arm swaps jumped over 0.2 rad in
one tick, none without the prefetch).

4. **RTC guidance** (``LEROBOT_CHUNK_RTC=1``, needs the prefetch). The prefetched
   inference is steered with LeRobot's Real-Time Chunking so that its first
   ``k`` steps -- the ticks the old chunk still runs while it infers, dropped
   at the swap -- follow the old chunk's remaining ``k`` actions. LeRobot 0.4.4
   guides only those steps; step ``k``, the first one executed, follows them
   through the model's own conditioning, so the jump shrinks but need not
   vanish (offline, 140 recorded swaps: median 0.22 -> 0.07 rad, max 1.13 ->
   0.21 rad). Flow-matching policies only (pi0, pi0.5, SmolVLA).
   ``LEROBOT_CHUNK_RTC_MAX_GUIDANCE`` overrides LeRobot's guidance clip.
5. **Seam blend** (``LEROBOT_CHUNK_SEAM_BLEND_TICKS=<m>``). At every swap, the
   arm joints of the new chunk get an offset that starts at the gap between the
   last sent command and the new chunk (position and per-tick velocity) and
   fades to zero along a quintic over at least ``m`` ticks -- longer when the
   gap is large, up to ``SEAM_BLEND_MAX_TICKS`` (or ``m`` if larger) and the
   ticks left in the chunk, aiming to move no joint more than
   ``SEAM_BLEND_MAX_STEP_RAD`` per tick; a gap too large for the cap exceeds
   it. Grippers and base are left alone. This
   is bumpless transfer: it removes the jump but still arrives on the new
   chunk's path, so it complements RTC rather than replacing it.
6. **Chunk smoothing** (``LEROBOT_CHUNK_SMOOTH_TICKS=<w>``, odd, >= 3). Every
   chunk, as it arrives and before anything else, gets its arm joints replaced by
   a centred moving average over ``w`` steps (the window shrinks at the ends; the
   first and the last step are kept as predicted). Grippers and base are left
   alone. Chunk-100 ACT checkpoints predict trajectories whose direction flips
   3-5x more often than the demonstrations (jitter in the prediction itself,
   not in execution); ``w=3`` brings that to the demonstrations' level with the
   L1 to the demonstration unchanged and the end pose untouched (offline,
   2026-10-08, experts and 11-stage chunk-100 alike). The seam blend (5) then
   sees the smoothed chunk.

The switches change nothing unless set: with none of the variables set this
module is not installed at all. When the chunk executor is installed with the
prefetch, the lead and the seam blend all off (e.g. for the execution log
alone), it reproduces ``select_action`` exactly.

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
RTC_VARIABLE = "LEROBOT_CHUNK_RTC"
RTC_MAX_GUIDANCE_VARIABLE = "LEROBOT_CHUNK_RTC_MAX_GUIDANCE"
SEAM_BLEND_TICKS_VARIABLE = "LEROBOT_CHUNK_SEAM_BLEND_TICKS"
SMOOTH_TICKS_VARIABLE = "LEROBOT_CHUNK_SMOOTH_TICKS"

SWITCH_VARIABLES = (
    PREFETCH_TICKS_VARIABLE,
    PREFETCH_INLINE_VARIABLE,
    BASE_LEAD_TICKS_VARIABLE,
    RTC_VARIABLE,
    RTC_MAX_GUIDANCE_VARIABLE,
    SEAM_BLEND_TICKS_VARIABLE,
    SMOOTH_TICKS_VARIABLE,
    replay_policy.DATASET_ENVIRONMENT_VARIABLE,
    replay_policy.EPISODES_ENVIRONMENT_VARIABLE,
    replay_policy.RUN_POLICY_ENVIRONMENT_VARIABLE,
    EXECUTION_LOG_VARIABLE,
)

BASE_ACTION_NAMES = ("x.vel", "theta.vel")
# Action names containing any of these are not arm joints for the seam blend.
NON_ARM_JOINT_MARKERS = ("carriage", "gripper")

# The seam blend lengthens its window until its offset moves no arm joint more
# than this per tick (0.1 rad/tick = 2.1 rad/s at 21 Hz; the 9/25 pi0 eval's
# ordinary ticks stayed under 0.062 rad in 99% of cases), up to the cap.
SEAM_BLEND_MAX_STEP_RAD = 0.1
SEAM_BLEND_MAX_TICKS = 20

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


def _read_positive_float(variable: str, problems: list[str]) -> float | None:
    value = os.getenv(variable, "").strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        problems.append(f"{variable}={value!r} is not a number; default kept")
        return None
    if not math.isfinite(number) or number <= 0:
        problems.append(f"{variable}={value!r} is not positive; default kept")
        return None
    return number


@dataclass(frozen=True)
class ChunkExecutionSettings:
    prefetch_ticks: int
    prefetch_inline: bool
    base_lead_ticks: int
    execution_log_path: str | None
    problems: tuple[str, ...] = field(default=())
    rtc: bool = False
    rtc_max_guidance: float | None = None
    seam_blend_ticks: int = 0
    smooth_ticks: int = 0

    @property
    def wants_executor(self) -> bool:
        return (
            self.prefetch_ticks > 0
            or self.base_lead_ticks > 0
            or self.seam_blend_ticks > 0
            or self.smooth_ticks > 0
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
    rtc = switch_on(os.getenv(RTC_VARIABLE, ""))
    if rtc and prefetch_ticks == 0:
        problems.append(
            f"{RTC_VARIABLE} does nothing without {PREFETCH_TICKS_VARIABLE}: it steers "
            "the prefetched inference toward the actions the old chunk runs meanwhile"
        )
    rtc_max_guidance = _read_positive_float(RTC_MAX_GUIDANCE_VARIABLE, problems)
    if rtc_max_guidance is not None and not rtc:
        problems.append(
            f"{RTC_MAX_GUIDANCE_VARIABLE} does nothing without {RTC_VARIABLE}"
        )
    seam_blend_ticks = _read_tick_count(SEAM_BLEND_TICKS_VARIABLE, problems)
    smooth_ticks = _read_tick_count(SMOOTH_TICKS_VARIABLE, problems)
    if smooth_ticks and (smooth_ticks < 3 or smooth_ticks % 2 == 0):
        problems.append(
            f"{SMOOTH_TICKS_VARIABLE}={smooth_ticks} must be an odd number >= 3 "
            "(a centred window); smoothing kept off"
        )
        smooth_ticks = 0
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
        rtc=rtc and prefetch_ticks > 0,
        rtc_max_guidance=rtc_max_guidance,
        seam_blend_ticks=seam_blend_ticks,
        smooth_ticks=smooth_ticks,
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


def smooth_arm_chunk(chunk, arm_indices, ticks: int):
    """Centred moving average over ``ticks`` steps on the arm joints of ``chunk``.

    ``chunk`` is ``[batch, steps, action_dim]`` (any device, any dtype).
    Returns ``(smoothed, max_abs_change)`` where ``max_abs_change`` is the largest
    change applied to any arm joint at any step, in the chunk's own units.

    The window is centred and shrinks symmetrically near the ends, and the
    first and the last step are kept exactly as predicted (``--fix-ends`` in the
    offline study): the seam blend reads the first step, and the end pose is what
    the next chunk continues from. ``ticks`` must be odd and >= 3; 1 or less
    returns the chunk unchanged. Grippers and base are untouched because they
    are not in ``arm_indices``.
    """
    if ticks is None or ticks <= 1 or not arm_indices:
        return chunk, 0.0
    if ticks % 2 == 0:
        raise ValueError(f"smoothing window must be odd, got {ticks}")
    steps = chunk.shape[1]
    if steps < 3:
        return chunk, 0.0
    half = ticks // 2
    arm = list(arm_indices)
    raw = chunk[:, :, arm]
    out = raw.clone()
    for step in range(1, steps - 1):
        reach = min(half, step, steps - 1 - step)
        out[:, step] = raw[:, step - reach : step + reach + 1].mean(dim=1)
    smoothed = chunk.clone()
    smoothed[:, :, arm] = out
    change = float((out - raw).abs().max().item()) if raw.numel() else 0.0
    return smoothed, change


def seam_blend_offsets(gap_position, gap_velocity, ticks: int):
    """Offsets to add to the first ``ticks`` executed steps of a new chunk.

    ``gap_position`` and ``gap_velocity`` are ``[dims]`` tensors: where the old
    command stream is minus where the new chunk would be, one tick before its
    first executed step (position, and per-tick change). The offset samples,
    once per tick, a quintic that starts at that gap with that slope and ends
    at zero with zero slope and curvature on step ``ticks - 1``; from there on
    the new chunk runs exactly. Sampled at few ticks, the first step carries
    only part of the velocity gap (about 3/4 of it at 4 ticks) -- position
    continuity is what the window size controls. Returns ``[ticks, dims]``; the
    last row is zero.
    """
    import torch

    s = torch.arange(1, ticks + 1, dtype=gap_position.dtype, device=gap_position.device)
    s = (s / ticks).unsqueeze(1)
    fade = 1 - 10 * s**3 + 15 * s**4 - 6 * s**5
    slope = s - 6 * s**3 + 8 * s**4 - 3 * s**5
    return gap_position.unsqueeze(0) * fade + ticks * gap_velocity.unsqueeze(0) * slope


def choose_seam_blend_ticks(
    gap_position,
    gap_velocity,
    minimum_ticks: int,
    maximum_ticks: int = SEAM_BLEND_MAX_TICKS,
    max_step: float = SEAM_BLEND_MAX_STEP_RAD,
) -> int:
    """Shortest window >= ``minimum_ticks`` whose offset moves no joint more than
    ``max_step`` per tick (inputs in the same unit as ``max_step``), searched up
    to ``maximum_ticks``. If none qualifies, the window with the smallest
    largest step: a longer window is not always gentler -- the velocity part of
    the offset grows with its length -- so the cap is no safe default.
    """
    import torch

    maximum_ticks = max(minimum_ticks, maximum_ticks)
    best_ticks, best_step = minimum_ticks, math.inf
    for ticks in range(minimum_ticks, maximum_ticks + 1):
        offsets = seam_blend_offsets(gap_position, gap_velocity, ticks)
        path = torch.cat([gap_position.unsqueeze(0), offsets])
        largest_step = float(path.diff(dim=0).abs().max())
        if largest_step <= max_step:
            return ticks
        if largest_step < best_step:
            best_ticks, best_step = ticks, largest_step
    return best_ticks


class _CompletedInference:
    """An inference already run in the loop thread (inline prefetch)."""

    def __init__(self, result, observation_tick: int, inference_seconds: float):
        self.result = result
        self.error = None
        self.observation_tick = observation_tick
        self.inference_seconds = inference_seconds
        self.overlap_prefix = None

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
        self.overlap_prefix = None
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
        # Swap ticks only, arm joints (grippers excluded), radians. The raw seam
        # is where the new chunk would have jumped; the sent seam is what was sent.
        "arm_seam_raw",
        "arm_seam_joint",
        "arm_seam_sent",
        "blend_ticks",
        # With the prefetch: largest gap between the new chunk's first k steps
        # (dropped at the swap) and what the old chunk sent at those ticks. Without
        # RTC it measures how far the two plans disagree; RTC steers it to zero.
        "prefix_gap",
        "arm_smooth_max",
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
        rtc: bool = False,
        seam_blend_ticks: int = 0,
        smooth_ticks: int = 0,
    ):
        self.policy = policy
        self.prefetch_ticks = prefetch_ticks
        self.prefetch_inline = prefetch_inline
        self.base_lead_ticks = base_lead_ticks
        self.rtc = rtc and prefetch_ticks > 0
        self.seam_blend_ticks = seam_blend_ticks
        self.smooth_ticks = smooth_ticks
        self.action_steps = policy.config.n_action_steps
        self._original_reset = policy.reset
        self._passes_observation_tick = isinstance(policy, replay_policy.ReplayPolicy)
        self._use_amp = bool(getattr(policy.config, "use_amp", False))
        self._stream = None
        self._slowest_inference_seconds: float | None = None
        # Rebound on every record_loop call (see begin_loop).
        self.base_indices: tuple[int, ...] = ()
        self.base_index_by_name: dict[str, int] = {}
        self.arm_indices: tuple[int, ...] = ()
        self._action_scale = None
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
        if self.rtc:
            rtc_config = getattr(self.policy.config, "rtc_config", None)
            guidance = getattr(rtc_config, "max_guidance_weight", None)
            parts.append(
                f"RTC guidance on the first {self.prefetch_ticks} steps "
                f"(max guidance {guidance})"
            )
        if self.seam_blend_ticks:
            if self.arm_indices:
                parts.append(
                    f"seam blend >= {self.seam_blend_ticks} ticks on "
                    f"{len(self.arm_indices)} arm joints"
                )
            else:
                parts.append("seam blend OFF (no arm joints in the action names)")
        if self.smooth_ticks:
            if self.arm_indices:
                parts.append(
                    f"chunk smoothing {self.smooth_ticks}-tick moving average on "
                    f"{len(self.arm_indices)} arm joints (ends fixed)"
                )
            else:
                parts.append("chunk smoothing OFF (no arm joints in the action names)")
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
        arm_indices=(),
    ):
        self.base_index_by_name = dict(base_index_by_name)
        self.base_indices = tuple(
            base_index_by_name[name]
            for name in BASE_ACTION_NAMES
            if name in base_index_by_name
        )
        self.arm_indices = tuple(arm_indices)
        self._action_scale = None
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
        # The last two actions select_action returned, for the seam blend.
        self._last_sent = None
        self._previous_sent = None
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

    def _predict(self, batch, observation_tick: int, prefix=None):
        if self._passes_observation_tick:
            return self.policy.predict_action_chunk(
                batch, observation_tick=observation_tick
            )
        if prefix is None:
            return self.policy.predict_action_chunk(batch)
        import torch

        # RTC's guidance takes a gradient with respect to the noisy actions, which
        # inference_mode forbids (both predict_action and the worker open it).
        # Leaving it is enough: the policy's parameters were frozen when RTC was
        # switched on, so nothing but that one elementwise step is recorded. The
        # prefix may itself be an inference tensor; cloning it here makes it a
        # normal one.
        with torch.inference_mode(False), torch.no_grad():
            prefix = prefix.clone()
            return self.policy.predict_action_chunk(
                batch,
                inference_delay=prefix.shape[1],
                prev_chunk_left_over=prefix,
                execution_horizon=prefix.shape[1],
            )

    def _overlap_prefix(self):
        """The ``k`` actions the current chunk sends while the next one infers.

        Normalized, as the policy produced them, with the arm joints as actually
        sent (after any seam blend) and the base channels as planned (before any
        lead). Step ``j`` of the next chunk is planned for the same tick as step
        ``j`` of this prefix.
        """
        step = self._current_elapsed + self._position
        prefix = self._current_chunk[:, step : step + self.prefetch_ticks].clone()
        if self.arm_indices:
            arms = list(self.arm_indices)
            sent = self._executing[:, self._position : self._position + prefix.shape[1]]
            prefix[:, :, arms] = sent[:, :, arms].to(
                device=prefix.device, dtype=prefix.dtype
            )
        return prefix

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

    def _start_worker(
        self, batch, observation_tick: int, prefix=None
    ) -> _BackgroundInference:
        predict = self._predict
        if prefix is not None:
            predict = functools.partial(self._predict, prefix=prefix)
        return _BackgroundInference(
            predict,
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
        if self.arm_indices:
            self._previous_sent = self._last_sent
            self._last_sent = action.detach().clone()
        if self.execution_log is not None:
            self._write_tick(tick, tick_record, action, chunk_step, lead_source)
        self._position += 1
        self._tick += 1
        return action

    def _launch(self, batch, tick: int, tick_record: dict) -> None:
        tick_record["launched"] = 1
        prefix = self._overlap_prefix()
        guide = prefix if self.rtc else None
        if self.prefetch_inline:
            start = time.perf_counter()
            try:
                result = self._predict(batch, tick, guide)
            except Exception as error:
                if guide is None:
                    raise
                # Same as a failed background inference: infer unguided instead.
                logger.warning(
                    f"Guided chunk inference failed ({error!r}); inferring "
                    "without RTC instead."
                )
                result = self._predict(batch, tick)
            seconds = time.perf_counter() - start
            tick_record["blocked_ms"] += seconds * 1e3
            self._blocked_seconds_since_swap += seconds
            self._pending = _CompletedInference(result, tick, seconds)
        else:
            self._pending = self._start_worker(batch, tick, guide)
        self._pending.overlap_prefix = prefix

    def _swap(self, batch, tick: int, tick_record: dict) -> None:
        result = None
        observation_tick = tick
        blocked_seconds = 0.0
        inference_seconds = None
        overlap_prefix = None
        pending = self._pending
        if pending is not None:
            result, blocked_seconds = self._collect(pending)
            self._pending = None
            if result is not None:
                observation_tick = pending.observation_tick
                inference_seconds = pending.inference_seconds
                overlap_prefix = pending.overlap_prefix
        if result is None:
            result, extra_blocked, inference_seconds = self._infer_now(batch, tick)
            blocked_seconds += extra_blocked
            observation_tick = tick

        elapsed = tick - observation_tick
        seam = self._seam_step(result, elapsed)
        arm_seam = self._install(result, observation_tick, elapsed)
        tick_record.update(arm_seam)
        if overlap_prefix is not None:
            tick_record["prefix_gap"] = self._arm_gap(
                result[:, : overlap_prefix.shape[1]], overlap_prefix
            )

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
                "arm_seam_raw": arm_seam.get("arm_seam_raw"),
                "arm_seam_sent": arm_seam.get("arm_seam_sent"),
                "blend_ticks": arm_seam.get("blend_ticks"),
                "prefix_gap": tick_record.get("prefix_gap"),
                "arm_smooth_max": arm_seam.get("arm_smooth_max"),
            }
        )

    def _install(self, chunk, observation_tick: int, elapsed: int) -> dict:
        """Make ``chunk`` the executing one; return the arm seam for the log."""
        smooth_max = None
        if self.smooth_ticks and self.arm_indices:
            # Before anything else: the seam blend, the overlap prefix and the
            # execution all read the smoothed chunk, so the log and the robot agree.
            raw = chunk
            chunk, _ = smooth_arm_chunk(chunk, self.arm_indices, self.smooth_ticks)
            smooth_max = self._arm_gap(chunk, raw)
        shifted = shift_base_channels(chunk, self.base_lead_ticks, self.base_indices)
        executing = shifted[:, : self.action_steps]
        if elapsed >= executing.shape[1]:
            logger.warning(
                f"Chunk arrived {elapsed} ticks after its observation, past its "
                f"{executing.shape[1]} actions; running only its last action."
            )
            elapsed = executing.shape[1] - 1
        executing = executing[:, elapsed:]
        arm_seam = {}
        if self._last_sent is not None and self.arm_indices:
            executing, arm_seam = self._blend_seam(executing)
        if smooth_max is not None:
            arm_seam = dict(arm_seam, arm_smooth_max=smooth_max)
        self._executing = executing
        self._current_chunk = chunk
        self._current_observation_tick = observation_tick
        self._current_elapsed = elapsed
        self._position = 0
        return arm_seam

    # ----- seam blend ------------------------------------------------------------

    def _scale(self, like):
        """Per-channel radians per normalized unit (the postprocessor is affine)."""
        import torch

        if self._action_scale is None:
            dimension = like.shape[-1]
            try:
                zeros = torch.zeros(1, dimension, dtype=like.dtype, device=like.device)
                ones = torch.ones(1, dimension, dtype=like.dtype, device=like.device)
                scale = self.postprocessor(ones.clone()) - self.postprocessor(
                    zeros.clone()
                )
                scale = torch.as_tensor(scale).detach().to("cpu", torch.float32)
                scale = scale.reshape(-1)
                if scale.shape[0] != dimension or not torch.isfinite(scale).all():
                    raise ValueError(f"unexpected scale {scale}")
            except Exception:
                logger.debug("No action scale from the postprocessor", exc_info=True)
                scale = torch.ones(dimension)
            self._action_scale = scale.abs()
        return self._action_scale

    def _arm_gap(self, first, second) -> float | None:
        """Largest |first - second| over arm joints and steps, in radians."""
        if not self.arm_indices:
            return None
        arms = list(self.arm_indices)
        scale = self._scale(first)[arms]
        difference = first[..., arms].float().cpu() - second[..., arms].float().cpu()
        return float((difference.abs() * scale).max())

    def _blend_seam(self, executing):
        """Offset the arm joints of ``executing`` so they continue the sent stream."""
        import torch

        arms = list(self.arm_indices)
        last = self._last_sent[0, arms].float().cpu()
        previous = (
            self._previous_sent[0, arms].float().cpu()
            if self._previous_sent is not None
            else last
        )
        head = executing[0, :2, arms].float().cpu()
        scale = self._scale(executing)[arms]
        new_first = head[0]
        new_velocity = (
            head[1] - head[0] if head.shape[0] > 1 else torch.zeros_like(last)
        )
        # Gap one tick before the first executed step, where the old stream sits.
        gap_position = last - (new_first - new_velocity)
        gap_velocity = (last - previous) - new_velocity
        raw = (new_first - last).abs() * scale
        arm_seam = {
            "arm_seam_raw": float(raw.max()),
            "arm_seam_joint": self.arm_indices[int(raw.argmax())],
            "arm_seam_sent": float(raw.max()),
            "blend_ticks": 0,
        }
        if not self.seam_blend_ticks:
            return executing, arm_seam
        available = executing.shape[1]
        minimum = min(self.seam_blend_ticks, available)
        ticks = choose_seam_blend_ticks(
            gap_position * scale,
            gap_velocity * scale,
            minimum,
            maximum_ticks=min(SEAM_BLEND_MAX_TICKS, available),
        )
        offsets = seam_blend_offsets(gap_position, gap_velocity, ticks)
        blended = executing.clone()
        blended[0, :ticks, arms] += offsets.to(
            device=blended.device, dtype=blended.dtype
        )
        sent = (blended[0, 0, arms].float().cpu() - last).abs() * scale
        arm_seam["arm_seam_sent"] = float(sent.max())
        arm_seam["blend_ticks"] = ticks
        return blended, arm_seam

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
        for key, label in (
            ("arm_seam_raw", "arm seam raw"),
            ("arm_seam_sent", "arm seam sent"),
            ("prefix_gap", "prefix gap"),
        ):
            values = [record[key] for record in later if record.get(key) is not None]
            if values:
                parts.append(
                    f"{label} median={statistics.median(values):.3f} "
                    f"max={max(values):.3f} rad"
                )
        blend = [record["blend_ticks"] for record in later if record.get("blend_ticks")]
        if blend:
            parts.append(
                f"blend ticks median={statistics.median(blend):g} max={max(blend)}"
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
    rtc = settings.rtc
    if rtc and not prefetch_ticks:
        notes.append("RTC off: it needs the prefetch, which is off")
        rtc = False
    if rtc and not (
        hasattr(policy, "init_rtc_processor") and hasattr(config, "rtc_config")
    ):
        notes.append(
            f"RTC off: policy type {policy_type!r} has no RTC (flow matching only)"
        )
        rtc = False
    if prefetch_ticks and settings.base_lead_ticks >= prefetch_ticks:
        notes.append(
            f"{BASE_LEAD_TICKS_VARIABLE}={settings.base_lead_ticks} is not below "
            f"{PREFETCH_TICKS_VARIABLE}={prefetch_ticks}: the lead's chunk tail starts "
            "no later than the next chunk's launch, so tail ticks hold until an "
            "inference lands (with the inline control too, when the lead is longer)"
        )

    if rtc:
        _enable_rtc(policy, prefetch_ticks, settings.rtc_max_guidance)

    executor = ChunkExecutor(
        policy,
        prefetch_ticks=prefetch_ticks,
        prefetch_inline=settings.prefetch_inline,
        base_lead_ticks=settings.base_lead_ticks,
        rtc=rtc,
        seam_blend_ticks=settings.seam_blend_ticks,
        smooth_ticks=settings.smooth_ticks,
    )
    # Instance attributes shadow the class methods; the class is left untouched.
    policy.select_action = executor.select_action
    policy.reset = executor.reset_policy
    policy._chunk_executor = executor
    return executor, "; ".join(notes)


# Mark a policy whose RTC we switched on: the rtc_config it had before, and the
# parameters we froze.
RTC_PREVIOUS_CONFIG_ATTRIBUTE = "_chunk_rtc_previous_config"
RTC_TRAINABLE_ATTRIBUTE = "_chunk_rtc_frozen_parameters"


def _enable_rtc(policy, prefix_ticks: int, max_guidance: float | None) -> None:
    """Switch LeRobot's RTC on for the executor's prefetched inferences only.

    The policy's own select_action refuses RTC, but the executor replaces it.
    The parameters are frozen so that RTC's gradient step records nothing but
    itself (and does not need the batch outside inference_mode).
    """
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    values = {"enabled": True, "execution_horizon": prefix_ticks}
    if max_guidance is not None:
        values["max_guidance_weight"] = max_guidance
    setattr(policy, RTC_PREVIOUS_CONFIG_ATTRIBUTE, policy.config.rtc_config)
    policy.config.rtc_config = RTCConfig(**values)
    policy.init_rtc_processor()
    parameters = getattr(policy, "parameters", None)
    if callable(parameters):
        trainable = [parameter for parameter in parameters() if parameter.requires_grad]
        setattr(policy, RTC_TRAINABLE_ATTRIBUTE, trainable)
        for parameter in trainable:
            parameter.requires_grad_(False)


def detach_chunk_executor(policy) -> None:
    """Undo ``attach_chunk_executor`` (used when a loop cannot be set up)."""
    for name in ("select_action", "reset", "_chunk_executor"):
        policy.__dict__.pop(name, None)
    if RTC_PREVIOUS_CONFIG_ATTRIBUTE in policy.__dict__:
        policy.config.rtc_config = policy.__dict__.pop(RTC_PREVIOUS_CONFIG_ATTRIBUTE)
        policy.init_rtc_processor()
        model = getattr(policy, "model", None)
        if policy.config.rtc_config is None and model is not None:
            # pi0's init_rtc_processor leaves the model's copy in place when the
            # config is None.
            model.rtc_processor = None
        for parameter in policy.__dict__.pop(RTC_TRAINABLE_ATTRIBUTE, ()):
            parameter.requires_grad_(True)


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


def _arm_joint_indices(dataset) -> tuple[int, ...]:
    """Action indices of arm joints: everything but base channels and grippers."""
    try:
        names = list(dataset.features["action"]["names"])
    except Exception:
        return ()
    return tuple(
        index
        for index, name in enumerate(names)
        if name not in BASE_ACTION_NAMES
        and not any(marker in name for marker in NON_ARM_JOINT_MARKERS)
    )


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
                arm_indices=_arm_joint_indices(dataset),
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
