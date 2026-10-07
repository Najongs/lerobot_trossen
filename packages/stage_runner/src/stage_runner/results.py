"""What a stage returns, and the closed vocabulary it may terminate with.

Stdlib only. aggregate.py and its unit test import this module to group runs by
`terminated_by`, and they have to keep working on any machine long after the
robot environment moves on -- importing lerobot or torch here would tie
re-aggregating an old events.jsonl to a working install of both.
"""

from dataclasses import dataclass, field

# The five values terminated_by may take. Closed on purpose: the aggregator
# buckets on this exact string, so a sixth value invented at a call site would
# become a silently separate bucket rather than an error. events.EventLog
# validates every emitted terminator against TERMINATED_BY_VALUES.
TERMINATED_BY_TIMEOUT: str = "timeout"
TERMINATED_BY_MANUAL: str = "manual"
TERMINATED_BY_STOP_RECORDING: str = "stop_recording"
TERMINATED_BY_RERECORD_REQUESTED: str = "rerecord_requested"
TERMINATED_BY_ERROR: str = "error"
# Added for the chain (config version 2). Each is a DISTINCT outcome, and the
# distinction is the measurement: the chain report counts automatic completions,
# manual completions and timeouts separately, because a chain that only got to
# stage 11 because a human pressed the right arrow ten times is not the result
# the run was trying to produce.
#
# `complete`  -- the completion monitor fired (progress held, output stalled,
#                past p10). The automatic signal.
# `reached`   -- a boundary reset arrived at the designated pose and held it.
# `not_reached` -- a boundary reset ran out its ceiling without arriving, or was
#                refused mid-entry. NOT folded into `timeout`: a stage that ran
#                long and a pose that was never reached need different answers,
#                and the second one means the arm is somewhere unplanned.
TERMINATED_BY_COMPLETE: str = "complete"
TERMINATED_BY_REACHED: str = "reached"
TERMINATED_BY_NOT_REACHED: str = "not_reached"
# `signal`: the stage was cut short by SIGTERM/SIGHUP -- cli.SignalLatch raises
# SystemExit(128 + signum) on the first delivery, and before 10/07 that landed
# in the same `error` bucket as a policy that blew up. The robot PC's ④' run
# (eval_najy_results_1007.md) showed the two were indistinguishable in events
# and in the chain report; only a log line told them apart. An operator's
# `kill`/dropped SSH is not a defect of the run, so it gets its own value.
TERMINATED_BY_SIGNAL: str = "signal"
TERMINATED_BY_VALUES: tuple[str, ...] = (
    TERMINATED_BY_TIMEOUT,
    TERMINATED_BY_MANUAL,
    TERMINATED_BY_STOP_RECORDING,
    TERMINATED_BY_RERECORD_REQUESTED,
    TERMINATED_BY_ERROR,
    TERMINATED_BY_SIGNAL,
    TERMINATED_BY_COMPLETE,
    TERMINATED_BY_REACHED,
    TERMINATED_BY_NOT_REACHED,
)


def signal_exit_name(error: BaseException) -> str | None:
    """'SIGTERM' / 'SIGHUP' when ``error`` is the SystemExit cli.SignalLatch raises
    for that signal (code 128 + signum), else None.

    Only the two signals the latch traps are recognised -- an ordinary
    ``sys.exit(1)`` or a ``SystemExit`` with a non-integer code stays an error.
    The check is on the exit code, not on the latch object, so the runner needs no
    handle on the latch and a test can inject ``SystemExit(129)`` directly.
    """
    import signal as _signal

    if not isinstance(error, SystemExit) or not isinstance(error.code, int):
        return None
    for name in ("SIGTERM", "SIGHUP"):
        number = getattr(_signal, name, None)
        if number is not None and error.code == 128 + int(number):
            return name
    return None

# A terminator TYPE that can be PLANNED but never REACHED. `completion` names
# the rule a chain policy stage ends by; the outcome is `complete`, `manual` or
# `timeout`. It appears on the stage_start event (as planned_terminator) and
# never on a stage_end, so it belongs in what EventLog accepts and NOT in what
# the aggregator buckets as an outcome.
PLANNED_ONLY_TERMINATORS: tuple[str, ...] = ("completion",)

# What EventLog.emit accepts in its `terminator` field: outcomes plus
# planned-only types. events.py validates against THIS; aggregate.py buckets on
# TERMINATED_BY_VALUES. Keeping them separate is what stops a planned-only value
# from becoming an outcome bucket nobody meant to create.
EMITTED_TERMINATOR_VALUES: tuple[str, ...] = (
    TERMINATED_BY_VALUES + PLANNED_ONLY_TERMINATORS
)

# Terminators that end the whole trial rather than just the stage: the operator
# pressed esc, the operator asked to re-record (we do not implement re-record,
# so a left arrow aborts the trial), or the stage raised. The runner breaks out
# of the stage loop on these and cli.main turns them into exit code 1.
#
# NOTHING IN THIS PACKAGE CLEARS THE KEYBOARD FLAGS, and this comment used to
# claim "the caller clears the flag". runner.run_trial and executors both state
# that events["rerecord_episode"] / events["stop_recording"] are never cleared,
# by design, so whoever wraps a P2 trial loop around run_trial has to decide for
# itself what a press between two trials means. A reader who believed the old
# sentence writes that loop without clearing, and trial 2 is then aborted before
# its first stage by a key pressed during trial 1.
ABORTING_TERMINATORS: frozenset[str] = frozenset(
    {
        TERMINATED_BY_STOP_RECORDING,
        TERMINATED_BY_RERECORD_REQUESTED,
        TERMINATED_BY_ERROR,
        TERMINATED_BY_SIGNAL,
    }
)

# reason values on the trial_end event. "aborted_empty" is separate from
# "aborted" because an abort before the first frame leaves nothing to save and
# the trial contributes no episode at all -- the aggregator must not count it
# as a zero-length chain.
TRIAL_REASON_COMPLETED: str = "completed"
TRIAL_REASON_ABORTED: str = "aborted"
TRIAL_REASON_ABORTED_EMPTY: str = "aborted_empty"
TRIAL_REASON_EXCEPTION: str = "exception"
# The trial was unwound by SIGTERM/SIGHUP (see TERMINATED_BY_SIGNAL). Kept apart
# from "exception" so the chain report does not count an operator's kill or a
# dropped SSH session as a crash; the base-stop and disconnect path is the same.
TRIAL_REASON_SIGNAL: str = "signal"
# A chain stage ended with a terminator its StageConfig.required_terminator does
# not allow: a policy stage timed out instead of completing, or a boundary reset
# never arrived. Separate from "aborted", which means a HUMAN stopped the run --
# the two need different answers, and a chain report that merged them would
# count a model failure as an operator abort. There is NO RETRY: the episode is
# saved with completed=false and the run ends.
TRIAL_REASON_CHAIN_FAILED: str = "chain_failed"


@dataclass(frozen=True)
class StageResult:
    """The outcome of one stage, as the executor measured it.

    terminated_by is one of TERMINATED_BY_VALUES; reason states the comparison
    that decided it, so a terminator inferred from elapsed time (record_loop
    self-clears events["exit_early"], leaving no other evidence) stays auditable
    in the JSONL rather than becoming an unexplained label.

    Success or failure is NOT here and never enters the runner: it comes from
    video plus the handwritten sheet and joins on wall_clock_iso in aggregate.py.
    """

    terminated_by: str
    elapsed_s: float
    frames: int
    reason: str = ""
    # Additive, JSON-safe numbers that explain the terminator: the progress
    # value and the stall length behind a `complete`, the ramp's T / dmax /
    # arrival error behind a `reached`, the clamped arm-ticks either way. It
    # lands in the stage_end event as `reason_detail` and is what
    # eval_chain_report tabulates.
    #
    # A DICT and not more fields, because the keys differ per executor and the
    # closed-vocabulary rule that applies to `terminated_by` would be wrong
    # here: the aggregator buckets on the terminator and must keep working when
    # a new executor reports a number it has never seen. `reason` stays the
    # human sentence; this is the numbers in it, machine-readable.
    detail: dict = field(default_factory=dict)

    @property
    def hertz(self) -> float:
        """Frames actually written per second during this stage.

        Measured from episode_buffer["size"] deltas and perf_counter, not from
        LEROBOT_LOOP_HZ_LOG: that is a module-global 30-frame window,
        unattributed to any stage, that discards a window after a >=1.0 s gap.
        Per-stage rate is what tells the operator whether the 30 fps assumption
        behind every timeout in the YAML held on this run.
        """
        if self.elapsed_s <= 0.0:
            return 0.0
        return self.frames / self.elapsed_s


@dataclass(frozen=True)
class TrialOutcome:
    """What run_trial hands back: the stage results AND whether the base stopped.

    ``base_is_stopped`` is deliberately NOT a field of StageResult. StageResult is
    what the executor measured about a stage; this is a fact about the hardware at
    a boundary, and it is the one fact the process exit code has to carry.

    run_trial used to return the results alone, so the only outcome that means
    "the base may still be driving" -- both stop_base paths failing at a boundary
    -- could not reach cli.main at all: the run exited 0 with trial_end
    reason="completed", indistinguishable from a clean run for any batch script
    that reads the exit status instead of parsing the JSONL. The failure IS in
    events.jsonl (transition.stop_base_path="failed"), and that is exactly the
    problem -- a batch loop that never opens the file never learns of it.

    False is STICKY across the trial: one failed boundary makes the whole trial's
    answer False even if every later stop worked, because a base left driving at
    stage 2 already happened and no later success undoes it.
    """

    results: list[StageResult]
    base_is_stopped: bool = True
    # The chain broke: a stage ended with a terminator its
    # StageConfig.required_terminator does not allow. Its own field for the same
    # reason base_is_stopped has one -- cli.main turns it into its own exit code,
    # so a batch script sees "the model did not finish the chain" without
    # parsing events.jsonl, and does NOT see it as the operator abort that exit
    # code 1 means.
    chain_failed: bool = False
    chain_failed_stage_id: str | None = None
