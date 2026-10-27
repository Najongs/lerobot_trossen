"""What a stage returns, and the closed vocabulary it may terminate with.

Stdlib only. aggregate.py and its unit test import this module to group runs by
`terminated_by`, and they have to keep working on any machine long after the
robot environment moves on -- importing lerobot or torch here would tie
re-aggregating an old events.jsonl to a working install of both.
"""

from dataclasses import dataclass

# The values terminated_by may take. Closed on purpose: the aggregator
# buckets on this exact string, so a sixth value invented at a call site would
# become a silently separate bucket rather than an error. events.EventLog
# validates every emitted terminator against TERMINATED_BY_VALUES.
TERMINATED_BY_TIMEOUT: str = "timeout"
TERMINATED_BY_MANUAL: str = "manual"
TERMINATED_BY_STOP_RECORDING: str = "stop_recording"
TERMINATED_BY_RERECORD_REQUESTED: str = "rerecord_requested"
TERMINATED_BY_ERROR: str = "error"
# The two ends of a base_replay / base_align stage that are not a timeout, a
# key or an exception. "converged": the stage reached what it was driving to (a
# settled replay, an aligned or heading-only align); the chain goes on.
# "base_failed": the stage stopped the base because carrying on was unsafe or
# meaningless (lost odometry, an e-stop, a refused command, an unusable
# estimate, a diverging correction); the outcome field of the stage's own event
# says which. Separate from "error" because nothing raised -- the stage decided.
TERMINATED_BY_CONVERGED: str = "converged"
TERMINATED_BY_BASE_FAILED: str = "base_failed"
# A policy stage the completion classifier ended before its timer
# (completion.py). Not aborting: the chain goes on to the next stage, as after
# a timeout. The same string as config.TERMINATOR_CLASSIFIER, which keeps the
# planned terminator on stage_start inside this vocabulary.
TERMINATED_BY_CLASSIFIER: str = "classifier"
# A policy stage a grasp terminator ended: wait_s after a gripper closed on
# something (completion.GripTracker). Same string as config.TERMINATOR_GRASP.
TERMINATED_BY_GRASP: str = "grasp"
TERMINATED_BY_VALUES: tuple[str, ...] = (
    TERMINATED_BY_TIMEOUT,
    TERMINATED_BY_MANUAL,
    TERMINATED_BY_STOP_RECORDING,
    TERMINATED_BY_RERECORD_REQUESTED,
    TERMINATED_BY_ERROR,
    TERMINATED_BY_CONVERGED,
    TERMINATED_BY_BASE_FAILED,
    TERMINATED_BY_CLASSIFIER,
    TERMINATED_BY_GRASP,
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
#
# base_failed aborts too: an align failure must not let the next policy run. A
# policy started from a base pose the align stage could not establish is not a
# measurement of that policy, and a base that stopped answering (odometry, an
# e-stop, refused commands) must not be handed to a policy that drives it.
ABORTING_TERMINATORS: frozenset[str] = frozenset(
    {
        TERMINATED_BY_STOP_RECORDING,
        TERMINATED_BY_RERECORD_REQUESTED,
        TERMINATED_BY_ERROR,
        TERMINATED_BY_BASE_FAILED,
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
    # The gripper goals this stage last SENT, for a stage whose commands never
    # reach the episode buffer -- a leader_reset stage records nothing, so
    # runner._recorded_gripper_goals has no row to read them back from. None for
    # a policy stage, whose goals the runner reads from the buffer as before.
    # The runner holds them through a following start pose move, so an object
    # the operator handed the robot stays held.
    gripper_goals: dict[str, float] | None = None

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
