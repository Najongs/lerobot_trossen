"""Stage-chaining metrics over one or more events.jsonl files, plus a labels CSV.

Stdlib only, and deliberately so: this module is re-run against OLD event logs
long after the run that produced them, on whatever machine happens to be free.
Importing lerobot or torch here would tie re-aggregating a 2026 log to a working
install of both. It imports nothing from this package except ``results``.

That guarantee runs through the package ``__init__``, which is why that file is
lazy (PEP 562 ``__getattr__``): an eager re-export list there executes before
this module's first line and pulled in draccus, lerobot and torch, which made
``python -m stage_runner.aggregate`` fail on any machine without the robot stack.
test_aggregate.py asserts the property so it cannot rot back.

It lives in the package rather than in ``scripts/`` for one reason: build item 18
requires a unit test, and a file under ``scripts/`` is not importable, so testing
it there means ``importlib.util.spec_from_file_location``. It gets a module CLI
instead::

    uv run python -m stage_runner.aggregate outputs/stage_runner/*/events.jsonl \\
        --labels labels.csv [--standalone standalone.csv] [--json summary.json]

``--labels`` is the operator's per-stage outcome sheet (run_id,stage_id,outcome).
``--standalone`` is a SECOND hand-filled file (STANDALONE_COLUMNS: stage_id,
trials,successes plus the provenance the denominator has to be compared on --
robot_type,fps,policy_path,measured_date), one row per stage evaluated ALONE
from a hand-set start state; without it the chaining loss ratio prints
UNAVAILABLE instead of a number, for the reason below.

THE BOUNDARY COSTS TWO THINGS AND THEY ARE LOGGED IN TWO HALVES. Since the
2026-09-08 safety fix the runner stops the base the instant the executor
returns, BEFORE the stage_end emit -- a base still driving must not wait on
bookkeeping. So the wall clock from stage_end to the next stage_start does NOT
contain stop_base any more: it is the RESIDUAL, and a 0.300 s stop_base shows up
in it as 0.000 s. This module therefore reports three numbers and never one:
``stop_base_s`` (the first half, read off the transition event), the residual
(the second half, derived from the two wall clocks) and ``boundary_cost_s``,
their per-boundary SUM -- which is the total the P1 question actually asks for.

WHY THIS IS THE ONE TESTED MODULE. It is the only hardware-free pure function in
the package, and when it is wrong it does not crash -- it prints a plausible
chain success rate and a plausible loss ratio that nobody can eyeball. The only
other way to discover an error is to count two dozen trials by hand.

SUCCESS NEVER ENTERS THE RUNNER. The machine records what it can measure (when a
stage started, how long it ran, how many frames it wrote, what ended it); a
person records success, failure and the failure phase from the video and the
handwritten sheet. They meet here, joined on (run_id, stage_id). This module
NEVER infers success from the event stream: a terminator firing means the stage
stopped, not that it worked.

The two evidence rules below are asymmetric on purpose, because both refuse to
guess in the same direction -- toward "we do not know":
  * a chain counts as successful through stage i only with an explicit success
    label on every stage up to i (a missing label is not a success);
  * the failure histogram counts a stage only on an explicit failure label (a
    missing label is not a failure either).

THE CHAINING LOSS RATIO NEEDS A NUMBER THIS DATA DOES NOT CONTAIN, and saying
so is this module's job. Its denominator is stage B's STANDALONE success rate --
B entered from a start state built by hand rather than from stage A's terminal
state -- and a chain trial only ever enters B out of A. Substituting B's in-chain
rate does not approximate it, it destroys it: with B scored only inside the
trials where A succeeded, numerator and denominator carry the same successes and
the ratio collapses to 1/P(A), a pure function of stage A that contains no
information about the seam whatsoever.

So this module computes what the chain DOES contain -- P(A), P(B|A), P(B|not A),
P(A and B), each with the counts that produced it and its exact definition
travelling in the output -- and reports the ratio as UNAVAILABLE until a
standalone P(B) is supplied with ``--standalone``, measured on the same robot,
fps and checkpoint as the chain trials it will be divided into (a denominator
from a different arm is not a denominator at all, so this module refuses rather
than dividing), naming the measurement that would produce it. It ships the
numerator's counts beside the denominator's for the same reason: "0.625" off
1 chain trial and off 40 are not the same claim. The first stage is the exception
that makes the whole thing tractable: A runs from the hand-set start state, so
the chain's own P(A) IS a standalone rate for it, and only B's has to be
acquired separately.

Getting the algebra right also DISSOLVES the vault's open question (§손실비,
"P(A→B)를 A 성공 조건부로 재야 ... 미확정") rather than deferring it:
P(B|A) / P_standalone(B) and P(A∧B) / (P(A) x P_standalone(B)) are the same
number, because P(A∧B) = P(A) x P(B|A). There is one ratio, not two variants.

That the metric was undecided while the runner was being written is exactly why
the runner writes raw events and pre-aggregates nothing: a metric defined next
month is recomputed over every log already on disk.
"""

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from stage_runner.results import TERMINATED_BY_VALUES

# The wire vocabulary this reader understands. Duplicated from events.py rather
# than imported from it on purpose: a log reader must keep reading logs written
# by versions of the writer it has never seen, so it pins the format it parses
# and gates on schema_version instead of following whatever the current writer
# happens to define.
EVENT_SCHEMA_VERSION: int = 1
EVENT_TRIAL_START: str = "trial_start"
EVENT_STAGE_START: str = "stage_start"
EVENT_STAGE_END: str = "stage_end"
EVENT_TRANSITION: str = "transition"
EVENT_TRIAL_END: str = "trial_end"

# Aggregator-side trial reason, not part of results.TRIAL_REASON_*: the runner
# can never write it. It means the log has no trial_end line at all -- the
# process was killed (SIGKILL, power loss) before it could finish the trial.
TRIAL_REASON_TRUNCATED: str = "truncated"

# [PROPOSAL -- UNCONFIRMED] the outcome vocabulary should be taken from the
# operator's actual handwriting once the first batch of trials exists, not
# invented here. Everything except this pair is unaffected by the choice: the
# JSONL does not carry outcomes at all.
OUTCOME_SUCCESS: str = "success"
OUTCOME_FAIL: str = "fail"
# Closed, and enforced by load_labels: every consumer here tests
# `== OUTCOME_SUCCESS`, so anything else IS a failure to them, and a typo would
# be counted as one silently.
OUTCOME_VALUES: tuple[str, ...] = (OUTCOME_SUCCESS, OUTCOME_FAIL)

LABEL_COLUMNS: tuple[str, ...] = ("run_id", "stage_id", "outcome")

# The standalone-rate file: the one input no chain log can ever contain. A row is
# one stage evaluated ALONE from a start state built by hand (for stage B of the
# chain, the operator poses the post-A state with the leader arm), scored the
# same way as a chain trial. COUNTS, not a rate, because a ratio whose
# denominator rests on 3 standalone trials and one that rests on 40 are not the
# same claim and the report has to be able to say which it is.
#
# THE PROVENANCE COLUMNS ARE REQUIRED, not optional metadata. This file's rate
# becomes the loss ratio's denominator by a join on stage_id alone, so without
# them a rate measured on another robot, at another fps, on last month's
# checkpoint divides silently into this month's chain numerator and the quotient
# still looks like a seam measurement. They are the axes heterogeneity() checks
# across the pooled chain trials that a CSV row can also carry -- checked here
# against that pool rather than left for a reader to notice. Not all of them:
# heterogeneity() also compares stage_runner_version, lerobot_version and
# config_version, which a standalone row has no column for, so a version drift
# between the two sessions is outside what this file can be checked on.
STANDALONE_COLUMNS: tuple[str, ...] = (
    "stage_id",
    "trials",
    "successes",
    "robot_type",
    "fps",
    "policy_path",
    "measured_date",
)

# How far the standalone measurement may sit from the chain trials in time before
# the report says so. A REVIEW TRIGGER, not a measured shelf life: nothing here
# knows how fast this robot drifts. Unlike robot_type/fps/policy_path this axis
# can never match exactly -- the standalone eval is a separate session by
# construction -- so it warns and the ratio is still computed, while a mismatch
# on the three exact axes refuses.
STANDALONE_AGE_WARNING_DAYS: int = 14

# Which mechanism stopped the base at a boundary. Duplicated from
# transitions.py rather than imported, for the same reason as the event names:
# this reader keeps reading logs written by writers it has never seen, and
# importing that module would pull lerobot into a stdlib-only file.
#
# THREE OUTCOMES, NOT TWO, and the difference is what the report has to carry:
#   primary  -- one get_observation + one send_action; the only path whose
#               stop_base_s is the cost of stopping the base
#   fallback -- the hold action failed (a dead camera is the likely cause) and
#               base.set_cmd_vel(0, 0) zeroed the base directly. THE BASE IS
#               STOPPED, but stop_base_s here is how long the failing call took
#   failed   -- neither path went through; the base may still be driving
STOP_PATH_PRIMARY: str = "primary"
STOP_PATH_FALLBACK: str = "fallback"
STOP_PATH_FAILED: str = "failed"
STOP_BASE_REASON_BY_PATH: dict[str, str] = {
    STOP_PATH_PRIMARY: "stop_base",
    STOP_PATH_FALLBACK: "stop_base_fallback",
    STOP_PATH_FAILED: "stop_base_failed",
}

# How far the logged `hertz` may sit from frames/elapsed_s before build_trial
# says so. Generous enough that a writer rounding to two decimals stays quiet,
# tight enough that a writer REDEFINING the field (an in-loop rate that excludes
# record_loop's entry cost, say) does not pass silently inside schema_version 1.
HERTZ_WARNING_TOLERANCE: float = 0.01


@dataclass(frozen=True)
class StageRecord:
    """One stage of one trial, reassembled from its stage_start/stage_end pair.

    A stage with a start and no end (the process died inside it) never becomes a
    StageRecord: it has no elapsed_s, no frame count and no terminator, so every
    metric here would have to invent one.
    """

    stage_id: str
    stage_index: int
    planned_terminator: str
    terminated_by: str
    reason: str
    elapsed_s: float
    frames: int
    start_frame_idx: int
    end_frame_idx: int
    start_wall_clock_iso: str
    end_wall_clock_iso: str

    @property
    def hertz(self) -> float:
        """Frames written per second, recomputed rather than read from the log.

        stage_end carries a `hertz` field and this ignores it, which keeps the
        reader working on a log written before that field existed. The two are
        compared once, in build_trial, which warns past HERTZ_WARNING_TOLERANCE
        -- that comparison is what makes this docstring checkable instead of a
        silent divergence. Without it a writer that redefined `hertz` inside
        schema_version 1 (the rename this reader already survived, gap_s ->
        stop_base_s, was exactly such an in-version change) would leave the same
        run's events.jsonl and its summary carrying two different rates for the
        same stage, with neither artifact saying so.

        0.0 when elapsed_s <= 0.0, and that 0.0 is missing data rather than a
        measured rate: a stage the operator aborted before it entered record_loop
        (executors._aborted_before_start) reports elapsed_s=0.0, frames=0.
        timing_summary drops those records from the rate statistics instead of
        averaging the zero in.
        """
        if self.elapsed_s <= 0.0:
            return 0.0
        return self.frames / self.elapsed_s


@dataclass(frozen=True)
class TrialRecord:
    """One run of the chain: one events.jsonl file, one dataset episode.

    THE THREE TIMING TUPLES ARE DIFFERENT QUANTITIES, and confusing them is the
    mistake this whole experiment cannot afford. The writer's order at a boundary
    is: executor returns -> stop_base -> stage_end emit -> transition emit ->
    next stage_start emit. That order is a SAFETY property (a base still driving
    must not wait on bookkeeping) and it decides what each of these contains:

    * ``stop_base_durations_s`` -- the ``stop_base_s`` field of each transition
      event: one get_observation plus one send_action, timed around the call. It
      runs BEFORE the stage_end emit, so it is the FIRST half of the boundary and
      is NOT inside the tuple below. It was called ``gap_s`` in the writer until
      2026-09-08 and reported here as the whole transition gap.
    * ``transition_gaps_s`` -- the stage_end -> next stage_start RESIDUAL,
      DERIVED here from ``stage_end.wall_clock_iso`` -> the next
      ``stage_start.wall_clock_iso``. Since stop_base already ran and finished
      before the stage_end emit, this covers only the two emits and the loop
      bookkeeping between them. Read alone it says a 0.300 s base stop cost
      0.000 s, which is why it is never printed alone.
    * ``boundary_costs_s`` -- the SUM of the two halves for one boundary, and the
      only one of the three that answers "what did crossing this seam cost".
      Paired per boundary rather than added as two means: they have different
      sample sizes (a boundary that leads nowhere has no residual, and a
      non-primary stop_base is not a timing sample at all), so summing the
      reported means would add numbers measured over different sets.

    WHAT THE SUM STILL EXCLUDES, so nobody reads it as the full
    command-to-command dead time: record_loop's ENTRY cost for the next stage
    (policy.reset, the first get_observation, the first ACT inference before the
    first send_action) happens AFTER stage_start is emitted and is therefore not
    in it; at the other end the last send_action of the previous stage precedes
    the executor's return by part of one control period. Capturing that needs a
    timestamp inside call_record_loop, which P1 does not have. Read the sum as a
    LOWER BOUND on the control gap, measured between events that are logged.

    THE THREE stop_base OUTCOMES ARE KEPT APART, because only one of them is a
    measurement and only one of them is a hazard:
    ``stop_base_durations_s`` holds the PRIMARY path alone;
    ``stop_base_fallback_stage_ids`` marks boundaries where the hold action
    failed and the base was zeroed directly instead (stopped, but the primary
    mechanism is broken -- a dead camera, most likely); and
    ``stop_base_failed_stage_ids`` marks the boundaries where nothing zeroed the
    base. Since 2026-09-08 the runner emits the transition event on EVERY
    boundary -- including after the last stage and after an abort, with
    ``to_stage_id: null`` -- precisely so that a run whose base was never stopped
    cannot be byte-identical to a clean one. The last boundary is the one that
    matters most: save_episode flushes video encoders for seconds afterwards
    while the base would still be holding the last policy velocity.
    """

    run_id: str
    started_iso: str
    ended_iso: str | None
    stages: tuple[StageRecord, ...]
    transition_gaps_s: tuple[float, ...]
    stop_base_durations_s: tuple[float, ...]
    stop_base_fallback_stage_ids: tuple[str, ...]
    stop_base_failed_stage_ids: tuple[str, ...]
    completed: bool
    end_reason: str
    # The per-boundary SUM of the two halves, for the boundaries where both were
    # logged: a crossing (so there is a residual) whose stop_base took the
    # primary path (so its duration is the cost of a stop and not of a failing
    # camera read). Defaulted to () because a hand-built TrialRecord in a test
    # carries no boundary at all; the fixture is where it is checked.
    boundary_costs_s: tuple[float, ...] = ()
    # PROVENANCE off trial_start, so a pooled summary can say what produced it.
    # Defaulted because a hand-built TrialRecord in a test has no trial_start to
    # read. Without these, one glob over outputs/ averages a cameraless mock run
    # (a clean 30 Hz by construction) into the same hertz_mean as the real ~20 Hz
    # loop, and pools trials recorded before and after a timer change into one
    # chain_success_rate. summarize() reports the disagreement rather than
    # refusing to aggregate: comparing two arms IS the point, doing it without
    # noticing is not.
    robot_type: str | None = None
    fps: int | None = None
    dataset_repo_id: str | None = None
    declared_stage_ids: tuple[str, ...] = ()
    policies: tuple[tuple[str, str], ...] = ()
    # THE CODE VERSIONS THAT PRODUCED THE RUN, also off trial_start. These are
    # the axis the other provenance fields cannot see: two trials can agree on
    # robot, fps and checkpoint and still have been measured by different
    # software. `gap_s` -> `stop_base_s` was exactly such a change -- it moved
    # what a boundary number MEANS without touching robot_type, fps or the
    # policy path -- so pooling across it silently averages two definitions into
    # one mean. Defaulted to None for a hand-built TrialRecord and for a log
    # written before the field existed; None is "unknown", and heterogeneity()
    # reports a pool that mixes known with unknown rather than assuming they
    # match.
    stage_runner_version: str | None = None
    lerobot_version: str | None = None
    config_version: int | None = None


def _seconds_between(earlier_iso: str, later_iso: str) -> float:
    """Seconds from one wall_clock_iso stamp to another, later minus earlier.

    Both stamps are local time with an explicit offset and millisecond
    resolution (events.wall_clock_iso), so datetime handles the arithmetic
    including a run that straddles a DST change.
    """
    return (
        datetime.fromisoformat(later_iso) - datetime.fromisoformat(earlier_iso)
    ).total_seconds()


def stop_base_path(event: Mapping[str, Any]) -> str:
    """Which mechanism stopped the base at this boundary.

    `stop_base_path` is authoritative, and the writer derives `reason` from it,
    so a log carrying the field is read off the field and an older one falls back
    to the reason string.

    stop_action IS NOT A USABLE PROXY for either, and reading it as one is the
    trap this function exists to close: it is null on the FALLBACK path, where
    the base was zeroed directly and only the arms went uncommanded, exactly as
    it is null on the failed path where the base may still be driving. Counting
    those two together would report a working safety fallback as a base left
    running -- or, worse the other way round, hide the failure inside a bucket
    the operator learns to ignore.
    """
    path = event.get("stop_base_path")
    if isinstance(path, str) and path in STOP_BASE_REASON_BY_PATH:
        return path
    reason = str(event.get("reason", ""))
    for candidate, candidate_reason in STOP_BASE_REASON_BY_PATH.items():
        if reason == candidate_reason:
            return candidate
    # Unknown wording from a newer writer is filed as a FAILURE on purpose. The
    # cost of over-reporting is one line in the report; the cost of the other
    # error is a base that drove through save_episode with nothing saying so.
    print(
        f"[warn] 모르는 stop_base reason / unrecognised transition reason "
        f"{reason!r} in run {event.get('run_id')!r} -- counted as a FAILED base "
        f"stop, which is the safe direction"
    )
    return STOP_PATH_FAILED


def load_events(path: Path) -> list[dict[str, Any]]:
    """Parse one events.jsonl into a list of dicts, oldest first.

    A malformed LAST line is dropped with a warning instead of raising: EventLog
    flushes per line, so the only way to get one is a kill landing mid-write,
    and that is precisely the run whose earlier stages we still want. A
    malformed interior line means the file is corrupt and raises.
    """
    events: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == len(raw_lines):
                print(
                    f"[warn] 마지막 줄이 잘림 / truncated final line, dropped: "
                    f"{path}:{line_number} ({error})"
                )
                continue
            raise ValueError(
                f"JSON 파싱 실패 / malformed event line: {path}:{line_number}"
            ) from error
        version = event.get("schema_version")
        if version != EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {version!r} != {EVENT_SCHEMA_VERSION} "
                f"({path}:{line_number}) -- 이 리더가 모르는 포맷 / "
                f"unknown event format, field meanings may have moved"
            )
        events.append(event)
    return events


def build_trial(events: Sequence[dict[str, Any]]) -> TrialRecord:
    """Reassemble one trial from its event stream.

    Tolerates a truncated stream: with no trial_end line the trial is
    completed=False, end_reason="truncated", and every stage that DID finish is
    still returned. That is the killed-process case, and its finished stages are
    real measurements.

    The stage_end -> stage_start RESIDUAL is DERIVED here rather than read from a
    field: wall clock from each stage_end to the next stage_start (TrialRecord
    says what that covers and what it leaves out -- since the 2026-09-08 safety
    fix stop_base has already finished by the stage_end emit, so it is NOT in
    there). That derivation is what makes this reader indifferent to a boundary
    that leads nowhere. The runner emits a transition event after EVERY stage --
    after the last one and after an abort too, carrying ``to_stage_id: null`` --
    so that a stop_base which silently failed cannot produce a log
    byte-identical to a clean run. No stage_start follows such a boundary, so it
    contributes a stop_base outcome and NO residual, and the residual sample
    still only ever contains boundaries that were crossed.

    THE TWO HALVES ARE PAIRED HERE, at the only place that sees them in order:
    the transition event carrying stop_base_s arrives between the stage_end and
    the stage_start whose difference is the residual, so a boundary's total is
    formed while both are in hand. Pairing downstream would mean adding one
    sample's mean to another's -- the two tuples have different lengths on
    purpose.
    """
    if not events:
        raise ValueError("이벤트가 없음 / empty event stream")
    start_events = [
        event for event in events if event.get("event") == EVENT_TRIAL_START
    ]
    if not start_events:
        raise ValueError(
            "trial_start 이벤트 없음 / no trial_start event -- "
            "not an events.jsonl written by stage_runner"
        )
    if len(start_events) > 1:
        raise ValueError(
            f"trial_start가 {len(start_events)}개 / multiple trial_start events -- "
            f"two runs concatenated into one file"
        )
    trial_start = start_events[0]
    run_id = str(trial_start["run_id"])
    run_ids = {str(event.get("run_id")) for event in events}
    if run_ids != {run_id}:
        raise ValueError(
            f"한 파일에 run_id가 섞임 / mixed run_ids in one file: {sorted(run_ids)}"
        )

    # Provenance, carried onto the TrialRecord so a pooled summary can say what
    # produced it. `or []` rather than a KeyError: these fields postdate nothing
    # but they are additive, and a trial that is missing them is still a trial
    # worth aggregating -- summarize() reports the disagreement it can see.
    policies = tuple(
        (str(entry.get("stage_id")), str(entry.get("policy_path")))
        for entry in (trial_start.get("policies") or [])
    )
    declared_stage_ids = tuple(
        str(stage_id) for stage_id in (trial_start.get("stage_ids") or [])
    )
    robot_type = trial_start.get("robot_type")
    dataset_repo_id = trial_start.get("dataset_repo_id")
    fps = trial_start.get("fps")
    # The three code versions. Read like the rest: absent means unknown, not a
    # mismatch, so a log written before the writer emitted them still aggregates.
    stage_runner_version = trial_start.get("stage_runner_version")
    lerobot_version = trial_start.get("lerobot_version")
    config_version = trial_start.get("config_version")

    stages: list[StageRecord] = []
    gaps: list[float] = []
    boundary_costs: list[float] = []
    stop_base_durations: list[float] = []
    stop_base_fallbacks: list[str] = []
    stop_base_failures: list[str] = []
    pending: dict[str, Any] | None = None
    # wall_clock_iso of the last stage_end, consumed by the next stage_start to
    # give the stage_end -> stage_start residual. Cleared as soon as it is
    # consumed -- or when a stage_start arrives with no stage_end since the
    # previous one -- so a residual is only ever measured across a boundary both
    # of whose ends were logged.
    previous_stage_end_iso: str | None = None
    # stop_base_s of the transition event for the boundary now open, and None
    # unless that boundary's stop took the PRIMARY path. The other two paths time
    # a call that failed rather than a stop, so adding one to a residual would
    # report a dead camera as the cost of crossing the seam. None also covers a
    # log written before the transition event existed: no first half, no total.
    pending_stop_base_s: float | None = None
    trial_end: dict[str, Any] | None = None
    for event in events:
        name = event.get("event")
        if name == EVENT_STAGE_START:
            if previous_stage_end_iso is not None:
                gap_s = _seconds_between(
                    previous_stage_end_iso, str(event["wall_clock_iso"])
                )
                if gap_s < 0.0:
                    # Wall clock, so a system clock step lands here rather than
                    # in a perf_counter. Kept, not dropped: a negative residual
                    # is evidence about the clock, and silently discarding it
                    # would leave a mean computed over an unstated subset.
                    print(
                        f"[warn] 음수 전환 잔여 / negative stage_end -> stage_start "
                        f"residual {gap_s:.3f}s in run {run_id} before stage "
                        f"{event.get('stage_id')!r} -- clock adjusted mid-run?"
                    )
                gaps.append(gap_s)
                # Both halves in hand: this boundary's total. Only here -- the
                # residual alone hides stop_base entirely now that it runs before
                # the stage_end emit.
                if pending_stop_base_s is not None:
                    boundary_costs.append(pending_stop_base_s + gap_s)
            previous_stage_end_iso = None
            pending_stop_base_s = None
            if pending is not None:
                # Two starts with no end between them: the first stage never
                # finished. Dropping it is the same rule as the truncated tail.
                print(
                    f"[warn] stage_end 없는 stage_start 무시 / dropping unfinished "
                    f"stage {pending.get('stage_id')!r} in run {run_id}"
                )
            pending = event
        elif name == EVENT_STAGE_END:
            if pending is None:
                raise ValueError(
                    f"stage_start 없는 stage_end / stage_end without stage_start "
                    f"in run {run_id}: {event.get('stage_id')!r}"
                )
            if pending.get("stage_id") != event.get("stage_id"):
                raise ValueError(
                    f"stage_start/stage_end 짝이 안 맞음 / mismatched stage pair in "
                    f"run {run_id}: {pending.get('stage_id')!r} -> "
                    f"{event.get('stage_id')!r}"
                )
            terminated_by = str(event.get("terminator"))
            if terminated_by not in TERMINATED_BY_VALUES:
                # Not fatal: an old or newer log may carry a value this build
                # does not know, and dropping the stage would silently shrink
                # every denominator. It buckets under its own raw string.
                print(
                    f"[warn] 모르는 terminator / unknown terminator "
                    f"{terminated_by!r} in run {run_id} stage "
                    f"{event.get('stage_id')!r}"
                )
            frames = int(event["frames"])
            elapsed_s = float(event["elapsed_s"])
            # The one free integrity check on the frame counter, and it costs a
            # subtraction: frame_idx is episode_buffer["size"] stamped at both
            # emits with only the executor in between, so their difference MUST
            # equal the frame count the executor measured. An add_frame outside
            # the executor, or a per-stage save_episode slipping in (save_episode
            # pops "size"), desynchronises them while frames_mean and hertz stay
            # perfectly plausible. A warning, not a raise -- an old log is still
            # worth reading.
            counted = int(event["frame_idx"]) - int(pending["frame_idx"])
            if counted != frames:
                print(
                    f"[warn] 프레임 수 불일치 / frame count mismatch in run "
                    f"{run_id} stage {event.get('stage_id')!r}: frame_idx delta "
                    f"{counted} != frames {frames} -- frame_idx no longer indexes "
                    f"the episode"
                )
            # StageRecord.hertz recomputes rather than reads; this is where the
            # two are actually compared, so that claim is a mechanism and not a
            # promise. A writer that redefines `hertz` without bumping
            # schema_version otherwise leaves the JSONL and the summary carrying
            # two different rates for the same stage.
            logged_hertz = event.get("hertz")
            if logged_hertz is not None and elapsed_s > 0.0:
                recomputed = frames / elapsed_s
                if abs(float(logged_hertz) - recomputed) > HERTZ_WARNING_TOLERANCE:
                    print(
                        f"[warn] hertz 불일치 / logged hertz {float(logged_hertz):.3f}"
                        f" != frames/elapsed_s {recomputed:.3f} in run {run_id} "
                        f"stage {event.get('stage_id')!r} -- the writer's `hertz` "
                        f"may mean something else now; this reader uses "
                        f"frames/elapsed_s"
                    )
            stages.append(
                StageRecord(
                    stage_id=str(event["stage_id"]),
                    stage_index=int(event["stage_index"]),
                    planned_terminator=str(pending.get("terminator")),
                    terminated_by=terminated_by,
                    reason=str(event.get("reason", "")),
                    elapsed_s=elapsed_s,
                    frames=frames,
                    start_frame_idx=int(pending["frame_idx"]),
                    end_frame_idx=int(event["frame_idx"]),
                    start_wall_clock_iso=str(pending["wall_clock_iso"]),
                    end_wall_clock_iso=str(event["wall_clock_iso"]),
                )
            )
            pending = None
            previous_stage_end_iso = str(event["wall_clock_iso"])
            # A new boundary opens here, so anything still held from the last one
            # (a transition whose stage_start never arrived) is stale.
            pending_stop_base_s = None
        elif name == EVENT_TRANSITION:
            # stop_base_s ONLY -- one get_observation plus one send_action, and
            # it has ALREADY HAPPENED by the time this event is written: the
            # runner stops the base as soon as the executor returns, before the
            # stage_end emit, so that a base still driving never waits on
            # bookkeeping. The stage_end -> stage_start residual derived above
            # therefore does not contain it; the two are summed per boundary into
            # boundary_costs_s, which is the number that answers what the seam
            # cost. This event covers neither the emits around it nor the next
            # stage's record_loop entry. The field was named
            # gap_s until 2026-09-08 and reported as the transition gap; no run
            # was ever recorded under that name, so nothing is read from it.
            if "stop_base_s" not in event:
                raise ValueError(
                    f"transition에 stop_base_s가 없음 / transition event without "
                    f"stop_base_s in run {run_id}: written before the 2026-09-08 "
                    f"rename, where the same measurement was called gap_s and was "
                    f"reported as the transition gap. No recorded run predates "
                    f"that rename; rename the key if you really have one."
                )
            # A boundary is written even when nothing was crossed (to_stage_id
            # null after the last stage and after an abort) and even when the
            # stop did not go through the primary path. ONLY THE PRIMARY PATH IS
            # A TIMING SAMPLE: on the other two, stop_base_s is how long the
            # failing get_observation took before it raised, so averaging it in
            # would report a stop that never happened as a slow one. The other
            # two are counted by identity instead, and timing_summary prints
            # them -- their only other trace is a logger.exception on a terminal
            # a batch script does not keep.
            boundary_stage_id = str(event.get("from_stage_id") or event.get("stage_id"))
            path = stop_base_path(event)
            if path == STOP_PATH_PRIMARY:
                stop_base_durations.append(float(event["stop_base_s"]))
                # Held for the stage_start that may follow, which is where the
                # residual becomes known and the boundary's total can be formed.
                # A boundary that leads nowhere never claims it, so it yields a
                # stop_base duration and no total -- there was nothing to cross.
                pending_stop_base_s = float(event["stop_base_s"])
            elif path == STOP_PATH_FALLBACK:
                stop_base_fallbacks.append(boundary_stage_id)
            else:
                stop_base_failures.append(boundary_stage_id)
        elif name == EVENT_TRIAL_END:
            trial_end = event
    if pending is not None:
        print(
            f"[warn] stage_end 없는 stage_start 무시 / dropping unfinished stage "
            f"{pending.get('stage_id')!r} in run {run_id}"
        )

    if trial_end is None:
        completed = False
        end_reason = TRIAL_REASON_TRUNCATED
        ended_iso = None
    else:
        completed = bool(trial_end.get("completed"))
        end_reason = str(trial_end.get("reason", ""))
        ended_iso = str(trial_end["wall_clock_iso"])
    return TrialRecord(
        run_id=run_id,
        started_iso=str(trial_start["wall_clock_iso"]),
        ended_iso=ended_iso,
        stages=tuple(stages),
        transition_gaps_s=tuple(gaps),
        boundary_costs_s=tuple(boundary_costs),
        stop_base_durations_s=tuple(stop_base_durations),
        stop_base_fallback_stage_ids=tuple(stop_base_fallbacks),
        stop_base_failed_stage_ids=tuple(stop_base_failures),
        completed=completed,
        end_reason=end_reason,
        robot_type=None if robot_type is None else str(robot_type),
        fps=None if fps is None else int(fps),
        dataset_repo_id=(None if dataset_repo_id is None else str(dataset_repo_id)),
        declared_stage_ids=declared_stage_ids,
        policies=policies,
        stage_runner_version=(
            None if stage_runner_version is None else str(stage_runner_version)
        ),
        lerobot_version=(None if lerobot_version is None else str(lerobot_version)),
        config_version=(None if config_version is None else int(config_version)),
    )


def load_trials(paths: Sequence[Path]) -> list[TrialRecord]:
    """One TrialRecord per run, over any number of events.jsonl files.

    Events are grouped by run_id rather than by file. A run directory holds one
    run, so normally that is one trial per path -- but concatenating old logs
    into one archive file is the obvious thing an operator does before handing
    over a batch, and grouping means that file reads exactly the same.

    The same run_id in two DIFFERENT files raises: globbing an outputs/ tree
    that still has a copied run directory in it would otherwise double every
    count in every metric, silently and plausibly.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    origin: dict[str, Path] = {}
    for path in paths:
        seen_in_this_file: set[str] = set()
        for event in load_events(path):
            run_id = str(event.get("run_id"))
            if run_id not in seen_in_this_file and run_id in grouped:
                raise ValueError(
                    f"run_id 중복 / duplicate run_id {run_id!r} across files: "
                    f"{origin[run_id]} and {path}"
                )
            seen_in_this_file.add(run_id)
            grouped.setdefault(run_id, []).append(event)
            origin.setdefault(run_id, path)
    return [build_trial(events) for events in grouped.values()]


def load_labels(path: Path) -> dict[tuple[str, str], str]:
    """Read the hand-filled labels CSV into {(run_id, stage_id): outcome}.

    Columns: run_id,stage_id,outcome[,failure_phase,note]. failure_phase and
    note are read by people, not by this module -- the failure histogram is
    derived from WHERE the first failing outcome sits in the stage order, so it
    needs no phase column. Both extra columns stay in the file because the
    3-way phase classification (inside the previous stage / at the transition /
    inside the next stage) is the operator's actual observation and has nowhere
    else to live.

    A repeated (run_id, stage_id) raises rather than last-one-wins: this file is
    typed by hand from a paper sheet, and a duplicated row is a transcription
    error whose silent resolution would change a metric.

    An outcome outside OUTCOME_VALUES raises for the same reason. Everything
    downstream tests `== OUTCOME_SUCCESS`, so a typo ("succes", "ok", "성공")
    would be filed as a FAILURE: chain_success_rate and average_sequence_length
    drop and the failure histogram gains a failure nobody recorded, with no
    warning anywhere. That breaks the module's own asymmetry rule, which is that
    neither missing nor unreadable evidence may ever land in the failure bucket.
    The vocabulary itself is still the spec's PROPOSAL -- when the operator's
    handwriting settles it, edit OUTCOME_SUCCESS / OUTCOME_FAIL and nothing
    else; this check only refuses values that are in neither bucket.

    THE CONTRACT HAS A SECOND FILE, and it has to be settled before the first
    batch of trials for the same reason the vocabulary does: runs accumulate
    against it. This file scores CHAIN trials, and no chain trial can give a
    later stage's STANDALONE success rate -- which is the chaining loss ratio's
    denominator. That number comes from load_standalone_rates' file
    (STANDALONE_COLUMNS), acquired by running the stage alone from a hand-set
    start state. Without it the ratio is reported UNAVAILABLE, and every
    trial recorded in the meantime stays un-analysable for the seam.
    """
    outcomes: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in LABEL_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"labels 파일에 열이 없음 / missing columns {missing} in {path} "
                f"(need {list(LABEL_COLUMNS)})"
            )
        for row_number, row in enumerate(reader, start=2):
            run_id = (row.get("run_id") or "").strip()
            stage_id = (row.get("stage_id") or "").strip()
            outcome = (row.get("outcome") or "").strip().lower()
            if not run_id and not stage_id and not outcome:
                continue
            if not run_id or not stage_id or not outcome:
                raise ValueError(
                    f"labels 행이 불완전 / incomplete label row {path}:{row_number}: "
                    f"{row!r}"
                )
            if outcome not in OUTCOME_VALUES:
                raise ValueError(
                    f"모르는 outcome 값 / unknown outcome {outcome!r} at "
                    f"{path}:{row_number} -- accepted values are "
                    f"{list(OUTCOME_VALUES)} (case-insensitive). "
                    f"오타는 실패로 집계되므로 거부 / a typo would otherwise be "
                    f"counted as a failure"
                )
            key = (run_id, stage_id)
            if key in outcomes:
                raise ValueError(
                    f"labels 중복 행 / duplicate label row for {key} at "
                    f"{path}:{row_number}"
                )
            outcomes[key] = outcome
    return outcomes


@dataclass(frozen=True)
class StandaloneRate:
    """One stage evaluated ALONE, from a start state built by hand.

    THE ONE INPUT NO CHAIN LOG CONTAINS. Stage B of the chain is only ever
    entered from stage A's terminal state, so nothing in events.jsonl is B's
    standalone success rate -- measuring it means posing the start state with the
    leader arm, running B by itself, and scoring it the same way. It is the
    denominator the chaining loss ratio needs, and until it exists that ratio is
    reported UNAVAILABLE rather than computed against B's in-chain rate.

    Counts rather than a rate, because a ratio resting on 3 standalone trials and
    one resting on 40 are different claims and the report has to say which.

    PROVENANCE IS REQUIRED, and it is required because of how this row is used:
    it is joined to the chain pool on stage_id ALONE and then becomes a
    denominator. A rate measured on another robot, at another fps, on another
    checkpoint divides into this month's numerator without anything looking
    wrong, and the quotient is still printed as "the share of B's standalone
    capability that survives the seam". These are the axes heterogeneity() checks
    across the pooled chain trials that a CSV row can carry too;
    standalone_comparability() checks them on this side and the ratio refuses on
    a mismatch rather than dividing. The version columns heterogeneity() also
    compares have no counterpart here -- see standalone_comparability().

    ``measured_date`` is the one axis that can never MATCH -- the standalone eval
    is a separate session by construction -- so it is reported and warned on
    (STANDALONE_AGE_WARNING_DAYS) instead of being required to agree.
    """

    stage_id: str
    trials: int
    successes: int
    # The arm this rate was measured on. Not defaulted: a default here would be a
    # guess about which robot produced a number that is about to become a
    # denominator, and the guess would be invisible in the report.
    robot_type: str
    fps: int
    # The checkpoint the stage ran, matched against the chain trials' policy_path
    # for the SAME stage. A rate from a different checkpoint measures a different
    # policy's standalone capability.
    policy_path: str
    # ISO date (YYYY-MM-DD) of the standalone session, validated on load.
    measured_date: str

    @property
    def rate(self) -> float:
        """Successes over trials; 0.0 for an empty file row, never a guess."""
        if self.trials <= 0:
            return 0.0
        return self.successes / self.trials


def load_standalone_rates(path: Path) -> dict[str, StandaloneRate]:
    """Read the standalone per-stage evaluation rows (STANDALONE_COLUMNS).

    stage_id,trials,successes plus the provenance the comparison needs:
    robot_type,fps,policy_path,measured_date.

    A SEPARATE ACQUISITION, not a column of the chain log, and deliberately its
    own file: mixing it into labels.csv would let a stage's standalone rate be
    typed on the same row as a chain trial's outcome, and those two numbers come
    from different runs of the robot.

    Rejects the same way load_labels does -- duplicate stage rows, successes
    above trials, non-positive trials -- because this file is also typed by hand
    and every one of those errors moves the ratio without moving anything else.

    THE PROVENANCE COLUMNS ARE REQUIRED AND VALIDATED HERE, not treated as
    optional notes. A blank robot_type is not "the same robot", it is an unknown
    one, and this row is on its way to becoming a denominator that will be
    divided into chain trials whose robot IS known. An unparseable measured_date
    is rejected for the same reason the outcome vocabulary is: the comparison
    downstream would silently skip an axis it was supposed to check.
    """
    rates: dict[str, StandaloneRate] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in STANDALONE_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"standalone 파일에 열이 없음 / missing columns {missing} in {path} "
                f"(need {list(STANDALONE_COLUMNS)})"
            )
        for row_number, row in enumerate(reader, start=2):
            values = {
                column: (row.get(column) or "").strip() for column in STANDALONE_COLUMNS
            }
            if not any(values.values()):
                continue
            blank = [column for column, value in values.items() if not value]
            if blank:
                raise ValueError(
                    f"standalone 행이 불완전 / incomplete standalone row "
                    f"{path}:{row_number}: missing {blank} -- 빈 provenance는 "
                    f"'같은 조건'이 아니라 '모르는 조건' / a blank provenance "
                    f"column is an unknown condition, not a matching one, and "
                    f"this row is about to become the loss ratio's denominator: "
                    f"{row!r}"
                )
            stage_id = values["stage_id"]
            try:
                trials = int(values["trials"])
                successes = int(values["successes"])
                fps = int(values["fps"])
            except ValueError as error:
                raise ValueError(
                    f"standalone 행의 수치가 정수가 아님 / non-integer counts at "
                    f"{path}:{row_number}: {row!r}"
                ) from error
            if trials <= 0 or successes < 0 or successes > trials:
                raise ValueError(
                    f"standalone 카운트가 불가능 / impossible counts at "
                    f"{path}:{row_number}: {successes} successes in {trials} trials"
                )
            if fps <= 0:
                raise ValueError(
                    f"standalone fps가 0 이하 / non-positive fps {fps} at "
                    f"{path}:{row_number}"
                )
            try:
                # Parsed, not just stored: the age comparison downstream would
                # otherwise skip this axis on a typo and report the standalone
                # rate as comparable when nobody checked when it was taken.
                date.fromisoformat(values["measured_date"])
            except ValueError as error:
                raise ValueError(
                    f"standalone measured_date가 ISO 날짜가 아님 / measured_date "
                    f"{values['measured_date']!r} is not an ISO date (YYYY-MM-DD) "
                    f"at {path}:{row_number}"
                ) from error
            if stage_id in rates:
                raise ValueError(
                    f"standalone 중복 행 / duplicate standalone row for "
                    f"{stage_id!r} at {path}:{row_number}"
                )
            rates[stage_id] = StandaloneRate(
                stage_id=stage_id,
                trials=trials,
                successes=successes,
                robot_type=values["robot_type"],
                fps=fps,
                policy_path=values["policy_path"],
                measured_date=values["measured_date"],
            )
    return rates


def stage_order(trials: Sequence[TrialRecord]) -> list[str]:
    """Stage ids in execution order, unioned across trials.

    Ordered by the stage_index the runner emitted, not by first appearance, so a
    trial that aborted at stage 1 does not reorder the axis every later metric
    is indexed by.
    """
    first_index: dict[str, int] = {}
    for trial in trials:
        for stage in trial.stages:
            known = first_index.get(stage.stage_id)
            if known is None or stage.stage_index < known:
                first_index[stage.stage_id] = stage.stage_index
    return [
        stage_id
        for stage_id, _ in sorted(
            first_index.items(), key=lambda item: (item[1], item[0])
        )
    ]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _spread(values: Sequence[float]) -> dict[str, Any]:
    """n / mean / min / max for one timing sample, zeros when it is empty."""
    return {
        "n": len(values),
        "mean": _mean(values),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def timing_summary(trials: Sequence[TrialRecord]) -> dict[str, Any]:
    """Per-stage wall clock, frame counts and achieved Hz. Needs no labels.

    Includes stages from INCOMPLETE trials: a stage that finished is a real
    measurement whether or not the trial that followed it did, and dropping it
    would quietly shrink the sample that the 30 fps assumption is checked
    against. `terminators` per stage is what makes an operator-aborted stage
    visible in that sample rather than hidden inside the mean.

    THE BOUNDARY IS REPORTED AS THREE KEYS, never one, because the writer stops
    the base BEFORE it emits stage_end (a base still driving must not wait on
    bookkeeping): `stop_base_s` is the first half,
    `stage_end_to_stage_start_residual_s` the residual left over after it (that
    key was `transition_gap_s` until 2026-09-08, when the safety fix moved
    stop_base out of the window it measures), and `boundary_cost_s`
    their per-boundary sum -- the total P1 asks for. The first two are disjoint,
    so neither is "of which" the other, and their sample sizes differ, which is
    why the total is paired in build_trial rather than added here.

    A STAGE THAT NEVER ENTERED record_loop IS NOT A SLOW STAGE, and this is the
    one number the whole trial batch is calibrated on. When the operator aborts
    between two stages the runner still emits a full stage_end for the
    stage that never started (executors._aborted_before_start returns
    elapsed_s=0.0, frames=0), and StageRecord.hertz reports 0.0 for it. Pooling
    that 0.0 is how nine clean 30 Hz stages plus one such record print
    hertz_mean=27.00 with hertz_min=0.00 -- and the MANDATORY FIRST-RUN PROCEDURE
    has the operator read exactly this number out of events.jsonl and set every
    timeout in the YAML from it. So the rate statistics are built from records
    with elapsed_s > 0.0, while `n` and `terminators` stay over all records and
    `not_run_n` says how many were dropped: missing data, made visible.

    `planned_vs_actual` is here because the terminator counts are INFERRED, not
    observed: record_loop self-clears events["exit_early"], so "manual" is a
    comparison against elapsed_s (record_adapter.classify_termination). A stage
    planned "manual" that ended "timeout" is a missed keypress that ran to the
    ceiling -- a corrupted measurement that the terminator histogram alone shows
    as an ordinary timeout.
    """
    stage_ids = stage_order(trials)
    per_stage: dict[str, Any] = {}
    for stage_id in stage_ids:
        records = [
            stage
            for trial in trials
            for stage in trial.stages
            if stage.stage_id == stage_id
        ]
        # elapsed_s > 0.0 IS the "this stage actually ran" test rather than a
        # threshold guess: _aborted_before_start is the only producer of an
        # exactly-0.0 elapsed_s, because everything else went through
        # perf_counter around a real record_loop call.
        measured = [stage for stage in records if stage.elapsed_s > 0.0]
        elapsed = [stage.elapsed_s for stage in measured]
        frames = [float(stage.frames) for stage in measured]
        hertz = [stage.hertz for stage in measured]
        terminators: dict[str, int] = {}
        planned_vs_actual: dict[str, int] = {}
        for record in records:
            terminators[record.terminated_by] = (
                terminators.get(record.terminated_by, 0) + 1
            )
            pair = f"{record.planned_terminator}->{record.terminated_by}"
            planned_vs_actual[pair] = planned_vs_actual.get(pair, 0) + 1
        per_stage[stage_id] = {
            "n": len(records),
            "n_measured": len(measured),
            "not_run_n": len(records) - len(measured),
            "elapsed_s_mean": _mean(elapsed),
            "elapsed_s_min": min(elapsed) if elapsed else 0.0,
            "elapsed_s_max": max(elapsed) if elapsed else 0.0,
            "frames_mean": _mean(frames),
            "hertz_mean": _mean(hertz),
            "hertz_min": min(hertz) if hertz else 0.0,
            "hertz_max": max(hertz) if hertz else 0.0,
            "terminators": terminators,
            "planned_vs_actual": planned_vs_actual,
        }
    gaps = [gap for trial in trials for gap in trial.transition_gaps_s]
    boundary_costs = [cost for trial in trials for cost in trial.boundary_costs_s]
    stop_base = [
        duration for trial in trials for duration in trial.stop_base_durations_s
    ]
    stop_base_fallback = [
        {"run_id": trial.run_id, "stage_id": stage_id}
        for trial in trials
        for stage_id in trial.stop_base_fallback_stage_ids
    ]
    stop_base_failed = [
        {"run_id": trial.run_id, "stage_id": stage_id}
        for trial in trials
        for stage_id in trial.stop_base_failed_stage_ids
    ]
    # The SECOND half of the boundary only: stage_end -> next stage_start.
    gap = _spread(gaps)
    # The caveat travels INSIDE the machine-readable output, not only in the
    # report string: a reader of summary.json sees the key and nothing else, and
    # what this number excludes is not small -- stop_base, which since the
    # 2026-09-08 safety fix has already finished before the stage_end emit, and
    # the next stage's whole record_loop entry, which is where the policy sits.
    gap["excludes"] = (
        "stop_base -- it runs the moment the executor returns, BEFORE the "
        "stage_end emit (a base still driving must not wait on bookkeeping), so "
        "a 0.300 s stop_base leaves this residual at 0.000 s; see "
        "boundary_cost_s for the total. Also excludes the next stage's "
        "record_loop entry (policy.reset, first get_observation, first "
        "inference, first send_action) and part of one control period before the "
        "executor returned. NOT MEASURED in P1: closing that needs a timestamp "
        "inside call_record_loop, which is P2 work"
    )
    # THE TOTAL, summed per boundary rather than as two means. The two halves
    # have different sample sizes by construction -- a boundary that leads
    # nowhere has no residual, and a non-primary stop_base is not a timing sample
    # -- so adding gap["mean"] to stop_base["mean"] would add numbers measured
    # over different sets of boundaries. build_trial pairs them where both exist.
    boundary_cost = _spread(boundary_costs)
    boundary_cost["definition"] = (
        "stop_base_s + the stage_end -> stage_start residual, for the boundaries "
        "where both halves were logged: a crossing (a stage_start followed) whose "
        "stop_base took the primary path. This is the total cost of crossing the "
        "seam that P1 asks for; the two halves are reported separately beside it "
        "because only one of them is a hardware cost"
    )
    boundary_cost["excludes"] = (
        "the next stage's record_loop entry and part of one control period before "
        "the executor returned -- the same unlogged remainder the residual "
        "excludes, so this is a LOWER BOUND"
    )
    # n on its own would read as "this is every boundary". It is not: these are
    # the boundaries where BOTH halves exist, and the shortfall against the
    # residual sample is boundaries whose stop_base fell back or failed.
    boundary_cost["crossed_boundary_n"] = len(gaps)
    boundary_cost["unpaired_boundary_n"] = len(gaps) - len(boundary_costs)
    return {
        "trial_count": len(trials),
        "completed_trial_count": sum(1 for trial in trials if trial.completed),
        "stage_ids": stage_ids,
        "per_stage": per_stage,
        # The second half of the boundary: the stage_end -> stage_start RESIDUAL.
        # RENAMED FROM transition_gap_s on 2026-09-08, and the rename is the
        # honest option rather than the disruptive one: before the safety fix
        # stop_base ran AFTER the stage_end emit, so the number under that key
        # WAS the whole seam, and afterwards -- same key, same shape, same units
        # -- it is the leftover with the hardware cost taken out. Keeping the
        # name would make an old-summary/new-summary diff read 0.032 -> 0.003 as
        # a tenfold speedup instead of a change of definition. A renamed key
        # shows up in that diff as one key gone and one arrived, which is what
        # actually happened. Read it with `excludes`, never alone; the total is
        # boundary_cost_s.
        "stage_end_to_stage_start_residual_s": gap,
        # The FIRST half, and not a component of the residual above: stop_base
        # runs before the stage_end emit, so the two are disjoint and their sum
        # is the boundary. Reported separately because only this half is a
        # hardware cost -- one get_observation, one send_action -- and only this
        # half is what the runner's transition event measures.
        "stop_base_s": _spread(stop_base),
        # The two halves added per boundary: what crossing the seam cost.
        "boundary_cost_s": boundary_cost,
        # Boundaries where the hold action failed and the base was zeroed
        # DIRECTLY instead. The base is stopped -- this is not a hazard -- but
        # the primary path is broken (a dead camera is the likely cause), and it
        # is invisible in stop_base_s by design, so it is reported here.
        "stop_base_fallback_n": len(stop_base_fallback),
        "stop_base_fallback": stop_base_fallback,
        # Boundaries where NOTHING zeroed the base, so it may have held the last
        # policy velocity through save_episode and the encoder flush. Counted,
        # never averaged into stop_base_s, and printed: the only other trace is a
        # logger.exception on a terminal that a batch script does not keep.
        "stop_base_failed_n": len(stop_base_failed),
        "stop_base_failed": stop_base_failed,
    }


def scored_trials(
    trials: Sequence[TrialRecord],
    labels: Mapping[tuple[str, str], str],
    stage_ids: Sequence[str],
) -> list[TrialRecord]:
    """The trials every success metric is computed over.

    A trial qualifies only if it is completed AND carries a label for the first
    stage. Both exclusions are deliberate:
      * not completed -- the runner aborted or was killed mid-chain, so the
        chain was never given the chance to fail on its own terms;
      * unlabelled -- nobody has watched the video yet, and an unlabelled trial
        is missing data, NOT a failed one.
    summarize() reports how many trials were excluded so the shrinkage is
    visible instead of showing up as a suspiciously small denominator.
    """
    if not stage_ids:
        return []
    first_stage = stage_ids[0]
    return [
        trial
        for trial in trials
        if trial.completed and (trial.run_id, first_stage) in labels
    ]


def per_stage_success_rate(
    trials: Sequence[TrialRecord],
    labels: Mapping[tuple[str, str], str],
    stage_ids: Sequence[str],
) -> dict[str, float]:
    """Success rate per stage, over the trials where THAT stage was labelled.

    The denominator is per stage, not the trial count, so a stage that never ran
    (the operator stopped the chain earlier) is not scored as a failure. A stage
    with no labels at all is absent from the dict rather than present as 0.0 --
    "not measured" and "measured as zero" must not print the same.
    """
    scored = scored_trials(trials, labels, stage_ids)
    rates: dict[str, float] = {}
    for stage_id in stage_ids:
        labelled = [trial for trial in scored if (trial.run_id, stage_id) in labels]
        if not labelled:
            continue
        successes = sum(
            1
            for trial in labelled
            if labels[(trial.run_id, stage_id)] == OUTCOME_SUCCESS
        )
        rates[stage_id] = successes / len(labelled)
    return rates


def _chain_length(
    trial: TrialRecord,
    labels: Mapping[tuple[str, str], str],
    stage_ids: Sequence[str],
) -> int:
    """How many leading stages this trial succeeded at, consecutively."""
    length = 0
    for stage_id in stage_ids:
        if labels.get((trial.run_id, stage_id)) != OUTCOME_SUCCESS:
            break
        length += 1
    return length


def chain_success_rate(
    trials: Sequence[TrialRecord],
    labels: Mapping[tuple[str, str], str],
    stage_ids: Sequence[str],
) -> list[float]:
    """CALVIN chain_sr: [P(stages 1..i all succeeded) for i in 1..N].

    Monotonically non-increasing by construction. Empty list when no trial
    qualifies -- an empty axis is not a row of zeros.
    """
    scored = scored_trials(trials, labels, stage_ids)
    if not scored:
        return []
    lengths = [_chain_length(trial, labels, stage_ids) for trial in scored]
    return [
        sum(1 for length in lengths if length >= prefix_length) / len(scored)
        for prefix_length in range(1, len(stage_ids) + 1)
    ]


def average_sequence_length(
    trials: Sequence[TrialRecord],
    labels: Mapping[tuple[str, str], str],
    stage_ids: Sequence[str],
) -> float:
    """Mean number of leading consecutive successes per trial (CALVIN avg len).

    Equals sum(chain_success_rate) for the same inputs; both are reported
    because the vector says WHERE the chain breaks and the scalar says how far
    it gets on average. Returns 0.0 when no trial qualifies -- read it together
    with summarize()'s scored_trial_count.
    """
    scored = scored_trials(trials, labels, stage_ids)
    if not scored:
        return 0.0
    return _mean([float(_chain_length(trial, labels, stage_ids)) for trial in scored])


def failure_histogram(
    trials: Sequence[TrialRecord],
    labels: Mapping[tuple[str, str], str],
    stage_ids: Sequence[str],
) -> dict[str, int]:
    """Per stage, the number of trials whose FIRST failure was there.

    mshab's subtask_fail_counts shape. Only an explicit failure label counts: a
    missing label is missing data, so a trial with no failing label anywhere
    contributes to no bucket at all rather than to the last stage's. Every
    stage_id is present, zeros included, because a stage that never failed is a
    result and must not vanish from the histogram.
    """
    counts = {stage_id: 0 for stage_id in stage_ids}
    for trial in scored_trials(trials, labels, stage_ids):
        for stage_id in stage_ids:
            outcome = labels.get((trial.run_id, stage_id))
            if outcome is not None and outcome != OUTCOME_SUCCESS:
                counts[stage_id] += 1
                break
    return counts


def missing_label_counts(
    trials: Sequence[TrialRecord],
    labels: Mapping[tuple[str, str], str],
    stage_ids: Sequence[str],
) -> dict[str, int]:
    """Per stage, how many SCORED trials carry no label row for it.

    A missing row is never a failure anywhere in this module -- that is the
    stated asymmetry rule -- which means it silently shrinks a denominator
    instead. One transcription gap off the paper sheet is enough to make the same
    stage report a high per-stage success rate and a low chain_sr in the same
    report, with nothing saying why. This is the counterpart of excluded_run_ids:
    a chain_sr depressed by data entry has to be distinguishable from one
    depressed by the robot.
    """
    scored = scored_trials(trials, labels, stage_ids)
    return {
        stage_id: sum(1 for trial in scored if (trial.run_id, stage_id) not in labels)
        for stage_id in stage_ids
    }


def _rate(numerator: int, denominator: int, definition: str) -> dict[str, Any]:
    """One observed probability, shipped with the counts and the definition.

    The definition travels WITH the number because every rate below differs from
    the others only in which trials are in its denominator, and a bare "0.50" in
    a report is exactly how an in-chain rate gets read as a standalone one. value
    is None, never 0.0, when the denominator is empty: "nothing to measure" and
    "measured as zero" must not print the same.
    """
    return {
        "value": (numerator / denominator) if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "definition": definition,
    }


def _unavailable(
    definition: str, reason: str, measurement: str | None = None
) -> dict[str, Any]:
    """A quantity this data cannot support, with why and what would fix it.

    Returning None alone would be honest but useless: the next reader recomputes
    it wrongly. The reason says what is missing and `measurement_required` says
    which acquisition produces it.
    """
    entry: dict[str, Any] = {
        "value": None,
        "available": False,
        "definition": definition,
        "unavailable_reason": reason,
    }
    if measurement is not None:
        entry["measurement_required"] = measurement
    return entry


# What the chain can and cannot say about the seam, written once here because
# both the summary and the report quote them.
LOSS_RATIO_DEFINITION: str = (
    "P(B | A succeeded) / P_standalone(B) -- the share of stage B's standalone "
    "capability that survives being entered from stage A's terminal state. "
    "1.0 = the seam costs nothing, below 1.0 = the chain loses at the seam. "
    "Identical to P(A and B) / (P(A) * P_standalone(B)), because "
    "P(A and B) = P(A) * P(B|A): there is ONE ratio here, not an A-conditional "
    "variant and an unconditional one."
)
ASSOCIATION_LIFT_DEFINITION: str = (
    "P(A and B) / (P(A) * P(B in chain)), over the trials labelled for BOTH "
    "stages -- the observed/expected association between the two labels on the "
    "SAME trials. NOT a chaining loss: its P(B) is the in-chain rate, so it "
    "compares B after a successful A against B after any A and says nothing "
    "about B standalone."
)
STANDALONE_MEASUREMENT: str = (
    "evaluate the stage ALONE: build its start state by hand with the leader arm "
    "instead of entering it from the previous stage, run N trials, score them the "
    "same way, and pass the counts with --standalone (STANDALONE_COLUMNS: "
    "stage_id,trials,successes,robot_type,fps,policy_path,measured_date) -- ON "
    "THE SAME ROBOT, fps and checkpoint as the chain trials it will be divided "
    "into, because that division is what the ratio is"
)


def standalone_comparability(
    rate: StandaloneRate, trials: Sequence[TrialRecord]
) -> dict[str, Any]:
    """Is this standalone rate measured under the chain pool's conditions?

    THE JOIN IS ON stage_id ALONE, which is the hole this closes. A standalone
    row is matched to the chain by one string and then becomes a denominator, so
    a rate taken on the mock robot, at 60 fps, or on the checkpoint before last
    divides into this pool's numerator and the quotient still prints as "the
    share of B's standalone capability that survives the seam". Nothing about the
    number looks wrong; only its provenance does, and nobody sees the provenance.

    The axes are the ones heterogeneity() checks ACROSS the pooled chain trials
    that a standalone CSV row can also carry -- this is the same check ACROSS THE
    JOIN instead. A pool that disagrees with itself is a mismatch too: there is
    then no single condition for the standalone rate to match, and picking the
    trial that agrees would be choosing the denominator to suit the answer.

    WHAT THIS CANNOT CHECK, stated so the pass is not read as more than it is:
    heterogeneity() also compares stage_runner_version, lerobot_version and
    config_version across the chain pool, and a standalone row records none of
    them. So a denominator measured by different software than the numerator
    passes here. That gap is the reason `measured_date` is reported even when it
    is inside the warning window -- the date is the only signal on this side that
    the two sessions could have been running different code.

    THE DATE IS TREATED DIFFERENTLY ON PURPOSE. robot_type, fps and policy_path
    can and must be equal; measured_date never can be, because the standalone
    eval is a separate session by construction. So it is reported always and
    warned past STANDALONE_AGE_WARNING_DAYS, and it never refuses.

    An axis the chain pool cannot answer (an old log carrying no provenance)
    lands in `unchecked` rather than in either bucket: unknown is not a mismatch,
    and it is not "checked and the same" either -- the same distinction between
    missing data and a measured zero that the rest of this module keeps.
    """
    checked: dict[str, Any] = {}
    mismatches: list[str] = []
    unchecked: list[str] = []
    warnings: list[str] = []

    def compare(axis: str, standalone_value: Any, pool_values: list[Any]) -> None:
        distinct: list[Any] = []
        for value in pool_values:
            if value is not None and value not in distinct:
                distinct.append(value)
        checked[axis] = {"standalone": standalone_value, "chain": distinct}
        if not distinct:
            unchecked.append(axis)
        elif distinct != [standalone_value]:
            mismatches.append(
                f"{axis}: standalone {standalone_value!r} vs chain {distinct!r}"
            )

    compare("robot_type", rate.robot_type, [trial.robot_type for trial in trials])
    compare("fps", rate.fps, [trial.fps for trial in trials])
    compare(
        "policy_path",
        rate.policy_path,
        [
            policy_path
            for trial in trials
            for stage_id, policy_path in trial.policies
            if stage_id == rate.stage_id
        ],
    )

    # The date axis: a separation, not an equality. It is measured against the
    # trials being divided (started_iso), never against today -- re-aggregating a
    # 2026 log in 2027 must not start warning about a pairing that was fine.
    chain_dates = sorted(
        {datetime.fromisoformat(trial.started_iso).date() for trial in trials}
    )
    measured_on = date.fromisoformat(rate.measured_date)
    if chain_dates:
        span = max(abs((measured_on - day).days) for day in chain_dates)
        checked["measured_date"] = {
            "standalone": rate.measured_date,
            "chain": [chain_dates[0].isoformat(), chain_dates[-1].isoformat()],
            "max_separation_days": span,
        }
        if span > STANDALONE_AGE_WARNING_DAYS:
            warnings.append(
                f"measured_date: the standalone eval is {span} days from the "
                f"furthest chain trial ({rate.measured_date} vs "
                f"{chain_dates[0].isoformat()}..{chain_dates[-1].isoformat()}), "
                f"past the {STANDALONE_AGE_WARNING_DAYS}-day review trigger -- "
                f"검토 필요 / check that nothing about the cell changed in between"
            )
    else:
        unchecked.append("measured_date")

    return {
        "comparable": not mismatches,
        "checked": checked,
        "mismatches": mismatches,
        "unchecked": unchecked,
        "warnings": warnings,
    }


def chaining_metrics(
    trials: Sequence[TrialRecord],
    labels: Mapping[tuple[str, str], str],
    stage_ids: Sequence[str],
    *,
    standalone_rates: Mapping[str, StandaloneRate] | None = None,
) -> dict[str, Any]:
    """The A -> B seam, as far as chain trials can measure it -- and no further.

    A is the first stage, B the second (P1 chains exactly two).

    WHAT THE CHAIN CONTAINS, all under "observed", each with its denominator:
    P(A), P(B in chain), P(B | A succeeded), P(B | A failed), P(A and B). Every
    one of them is a count over trials that were actually run, and none of them
    needs a definitional choice.

    WHAT IT DOES NOT CONTAIN is B's STANDALONE success rate, because a chain
    trial only ever enters B out of A. That number is the loss ratio's
    denominator, and substituting B's in-chain rate for it is not an
    approximation: in the normal case (B is only scored where A succeeded) the
    numerator and the denominator then carry the same successes, the ratio
    reduces to 1/P(A) exactly, and it reports "chaining helps" the worse stage A
    gets. So the ratio is UNAVAILABLE until standalone_rates carries B, and the
    output says which measurement would produce it.

    The FIRST stage is the exception, and it is what makes the ratio reachable at
    all: A runs from the hand-set start state, which IS the standalone condition,
    so the chain's own P(A) needs no separate acquisition.

    A SUPPLIED DENOMINATOR IS NOT AUTOMATICALLY THIS POOL'S DENOMINATOR. The join
    is on stage_id alone, so standalone_comparability() checks the row's
    provenance against the pooled chain trials (robot_type, fps, the checkpoint
    for that stage; the date is reported and warned on, since it can never
    match) and the ratio is UNAVAILABLE on a mismatch rather than dividing two
    conditions into each other. Both sides then ship their COUNTS, because the
    ratio is quoted far from here and a rate on 1 trial and a rate on 40 print
    identically.

    THE ASSOCIATION LIFT is the only seam-flavoured number the chain alone can
    produce, and it is reported ONLY when it is not degenerate. It needs B
    successes in both A-cells; when every B success sits inside an A success it
    equals 1/P(A) identically and when there are no A failures at all it equals
    1.0 identically. Both cases are arithmetic, not evidence, so they are
    reported as unavailable with the identity named -- a plausible number nobody
    can eyeball is the exact failure this module's unit test exists to prevent.

    Missing labels follow the module's asymmetry rule everywhere here: each rate
    is over the trials labelled FOR ITS OWN stages, so an untyped row shrinks a
    denominator instead of counting as a failure. summarize() reports how many.
    """
    rates = dict(standalone_rates or {})
    first_stage = stage_ids[0] if stage_ids else None
    second_stage = stage_ids[1] if len(stage_ids) > 1 else None
    metrics: dict[str, Any] = {
        "first_stage": first_stage,
        "second_stage": second_stage,
        "scored_trial_count": 0,
        "observed": {},
        "chaining_loss_ratio": _unavailable(
            LOSS_RATIO_DEFINITION,
            "체인이 두 단계 미만 / fewer than two stages in the pooled logs",
            STANDALONE_MEASUREMENT,
        ),
        "second_stage_association_lift": _unavailable(
            ASSOCIATION_LIFT_DEFINITION,
            "체인이 두 단계 미만 / fewer than two stages in the pooled logs",
        ),
    }
    if second_stage is None:
        return metrics

    scored = scored_trials(trials, labels, stage_ids)
    metrics["scored_trial_count"] = len(scored)
    if not scored:
        empty = "채점 대상 시행 0건 / no trial is both completed and labelled"
        metrics["chaining_loss_ratio"]["unavailable_reason"] = empty
        metrics["second_stage_association_lift"]["unavailable_reason"] = empty
        return metrics

    def outcome(trial: TrialRecord, stage_id: str) -> str | None:
        return labels.get((trial.run_id, stage_id))

    def succeeded(trial: TrialRecord, stage_id: str) -> bool:
        return outcome(trial, stage_id) == OUTCOME_SUCCESS

    first_ok = [trial for trial in scored if succeeded(trial, first_stage)]
    second_labelled = [
        trial for trial in scored if outcome(trial, second_stage) is not None
    ]
    second_ok = [trial for trial in second_labelled if succeeded(trial, second_stage)]
    after_success = [
        trial for trial in second_labelled if succeeded(trial, first_stage)
    ]
    after_success_ok = [
        trial for trial in after_success if succeeded(trial, second_stage)
    ]
    after_failure = [
        trial
        for trial in second_labelled
        if outcome(trial, first_stage) is not None and not succeeded(trial, first_stage)
    ]
    after_failure_ok = [
        trial for trial in after_failure if succeeded(trial, second_stage)
    ]
    both_labelled = [
        trial
        for trial in scored
        if outcome(trial, first_stage) is not None
        and outcome(trial, second_stage) is not None
    ]
    both_ok = [
        trial
        for trial in both_labelled
        if succeeded(trial, first_stage) and succeeded(trial, second_stage)
    ]

    metrics["observed"] = {
        "first_stage_success_rate": _rate(
            len(first_ok),
            len(scored),
            f"P({first_stage}) over scored trials. The chain starts this stage "
            f"from the hand-set start state, so this IS a standalone rate for "
            f"the FIRST stage -- which is why only later stages need a separate "
            f"standalone acquisition.",
        ),
        "second_stage_success_rate_in_chain": _rate(
            len(second_ok),
            len(second_labelled),
            f"P({second_stage} in chain) over the trials labelled for it. NOT a "
            f"standalone rate: every one of these entered {second_stage} from "
            f"{first_stage}'s terminal state, so this number already contains the "
            f"seam effect and must never be used as the loss ratio's denominator.",
        ),
        "second_stage_success_rate_given_first_success": _rate(
            len(after_success_ok),
            len(after_success),
            f"P({second_stage} | {first_stage} succeeded) -- the numerator of "
            f"the chaining loss ratio. Only trials where {second_stage} was "
            f"actually entered from a real successful {first_stage}.",
        ),
        "second_stage_success_rate_given_first_failure": _rate(
            len(after_failure_ok),
            len(after_failure),
            f"P({second_stage} | {first_stage} failed). Usually an empty "
            f"denominator -- a failed pour leaves nothing to dispose of, so the "
            f"chain normally does not score {second_stage} after a failed "
            f"{first_stage}. When it is NOT empty, comparing it against the line "
            f"above is the only seam signal the chain carries by itself.",
        ),
        "joint_success_rate": _rate(
            len(both_ok),
            len(both_labelled),
            f"P({first_stage} and {second_stage}) over the trials labelled for "
            f"BOTH stages.",
        ),
    }

    # --- the loss ratio: needs the standalone denominator ------------------
    conditional = metrics["observed"]["second_stage_success_rate_given_first_success"][
        "value"
    ]
    standalone = rates.get(second_stage)
    comparability = (
        None if standalone is None else standalone_comparability(standalone, scored)
    )
    if standalone is None:
        metrics["chaining_loss_ratio"]["unavailable_reason"] = (
            f"{second_stage!r}의 standalone 성공률이 없다 / no standalone success "
            f"rate for {second_stage!r}. Chain trials only ever enter it from "
            f"{first_stage}'s terminal state, so the log cannot contain one, and "
            f"substituting the in-chain rate collapses the ratio to 1/P("
            f"{first_stage}) -- a number about the first stage, not about the seam."
        )
    elif comparability is not None and not comparability["comparable"]:
        # REFUSED, not flagged. The other unavailable reasons here are all "the
        # data does not contain this"; so is this one. A denominator measured on
        # a different robot, fps or checkpoint is a rate for a different cell,
        # and dividing by it produces a number that is about no experiment that
        # was run -- while looking exactly like one that was.
        metrics["chaining_loss_ratio"]["unavailable_reason"] = (
            f"standalone 측정 조건이 체인과 다르다 / the standalone rate for "
            f"{second_stage!r} was not measured under this chain pool's "
            f"conditions: {'; '.join(comparability['mismatches'])}. 다른 조건의 "
            f"분모로 나누면 이음매가 아니라 조건 차이를 재게 된다 / dividing by it "
            f"measures the difference between the two conditions, not the seam."
        )
        metrics["chaining_loss_ratio"]["comparability"] = comparability
    elif standalone.rate == 0.0:
        metrics["chaining_loss_ratio"]["unavailable_reason"] = (
            f"standalone P({second_stage}) = 0 ({standalone.successes}/"
            f"{standalone.trials}) -- 분모가 0 / zero denominator"
        )
    elif conditional is None:
        metrics["chaining_loss_ratio"]["unavailable_reason"] = (
            f"{first_stage!r}가 성공하고 {second_stage!r}가 라벨된 시행이 0건 / no "
            f"scored trial where {first_stage!r} succeeded and {second_stage!r} "
            f"was labelled"
        )
    else:
        # BOTH SIDES SHIP THEIR COUNTS. The denominator's were already here, so
        # "32/40 trials run alone" travelled with the number while the numerator
        # stayed a bare rate -- and the numerator is the side that comes from
        # this run: a 0.625 resting on 1 chain trial and one resting on 40 are
        # not the same claim, and the seam question turns on exactly that.
        numerator_entry = metrics["observed"][
            "second_stage_success_rate_given_first_success"
        ]
        metrics["chaining_loss_ratio"] = {
            "value": conditional / standalone.rate,
            "available": True,
            "definition": LOSS_RATIO_DEFINITION,
            "numerator_value": conditional,
            "numerator_successes": numerator_entry["numerator"],
            "numerator_trials": numerator_entry["denominator"],
            "numerator_source": (
                f"in-chain: {numerator_entry['numerator']}/"
                f"{numerator_entry['denominator']} trials where {first_stage} "
                f"succeeded and {second_stage} was labelled"
            ),
            "denominator_value": standalone.rate,
            "denominator_successes": standalone.successes,
            "denominator_trials": standalone.trials,
            "denominator_source": (
                f"standalone evaluation of {second_stage!r}: "
                f"{standalone.successes}/{standalone.trials} trials run alone on "
                f"{standalone.robot_type} at {standalone.fps} fps, policy "
                f"{standalone.policy_path}, measured {standalone.measured_date}"
            ),
            # Kept on the AVAILABLE entry too, not only on the refusal: the axes
            # the chain pool could not answer, and the date separation, are why
            # this quotient is quotable -- or how far it can be trusted.
            "comparability": comparability,
        }

    # --- the association lift: chain-only, and usually degenerate ----------
    total_both = len(both_labelled)
    first_ok_both = [trial for trial in both_labelled if succeeded(trial, first_stage)]
    second_ok_both = [
        trial for trial in both_labelled if succeeded(trial, second_stage)
    ]
    cell_success_success = len(both_ok)
    cell_failure_success = len(second_ok_both) - cell_success_success
    if total_both == 0:
        metrics["second_stage_association_lift"]["unavailable_reason"] = (
            "두 단계 모두 라벨된 시행이 0건 / no trial is labelled for both stages"
        )
    elif not first_ok_both or not second_ok_both:
        metrics["second_stage_association_lift"]["unavailable_reason"] = (
            f"성공 사례가 없다 / no success to associate: {first_stage} succeeded "
            f"in {len(first_ok_both)} and {second_stage} in {len(second_ok_both)} "
            f"of {total_both} doubly-labelled trials"
        )
    elif cell_failure_success == 0:
        # Every B success sits inside an A success, which is the NORMAL shape of
        # a chain (a failed first stage leaves nothing for the second to do). The
        # lift is then P(B)/(P(A)*P(B)) = 1/P(A) exactly -- no seam information at
        # all -- and printing it is how "chaining beats independence by 2x" gets
        # read off a stage-A failure rate.
        probability_first = len(first_ok_both) / total_both
        identity = (
            "1.000 (P(A) = 1.0, 즉 A 실패 시행 자체가 없다)"
            if len(first_ok_both) == total_both
            else f"1/P({first_stage}) = {1.0 / probability_first:.3f}"
        )
        metrics["second_stage_association_lift"]["unavailable_reason"] = (
            f"{first_stage}가 실패한 시행 중 {second_stage} 성공이 0건 -- lift는 "
            f"{identity}로 대수적으로 고정돼 이음매 정보가 없다 / degenerate: with "
            f"every {second_stage} success inside a {first_stage} success the lift "
            f"is that value by construction, whatever the seam does"
        )
    elif cell_success_success == 0:
        metrics["second_stage_association_lift"]["unavailable_reason"] = (
            f"{first_stage} 성공 시행 중 {second_stage} 성공이 0건 -- lift는 0.0으로 "
            f"고정 / degenerate: the joint cell is empty"
        )
    else:
        probability_first = len(first_ok_both) / total_both
        probability_second = len(second_ok_both) / total_both
        probability_joint = cell_success_success / total_both
        metrics["second_stage_association_lift"] = {
            "value": probability_joint / (probability_first * probability_second),
            "available": True,
            "definition": ASSOCIATION_LIFT_DEFINITION,
            "trials": total_both,
            "cells": {
                "first_ok_second_ok": cell_success_success,
                "first_failed_second_ok": cell_failure_success,
            },
        }
    return metrics


def heterogeneity(trials: Sequence[TrialRecord]) -> dict[str, list[Any]]:
    """Provenance fields whose value is not the same across the pooled trials.

    Refusing to aggregate would be wrong -- comparing two arms IS the point --
    but pooling silently is how a cameraless mock run at a construction-clean
    30 Hz lands in the same hertz_mean as the real ~20 Hz loop, and how trials
    from before and after a timer change land in one chain_success_rate. Empty
    dict when every trial agrees, so an empty result means "checked and the same"
    rather than "not checked".

    `declared_stage_ids` (the chain the CONFIG declared, not what ran) is the
    entry that catches the worst pool: chain_success_rate is indexed by the union
    of stage ids across trials, so pooling a 3-stage run with a 2-stage one
    scores the shorter trial as having failed a stage it was never configured to
    run. An aborted trial does NOT trip this -- it declared the full chain.

    THE THREE VERSION COLUMNS catch what the other four cannot see: the same
    robot, fps and checkpoints measured by different software. The rename this
    reader already survived inside schema_version 1 (`gap_s` -> `stop_base_s`)
    changed what a boundary number MEANS while every other column stayed
    identical, so without these a pool spanning that change reports one mean over
    two definitions and says nothing. They are plain scalars and need none of the
    list flattening `declared_stage_ids` and `policy_paths` use.
    """
    columns: dict[str, list[Any]] = {
        "robot_type": [trial.robot_type for trial in trials],
        "fps": [trial.fps for trial in trials],
        "declared_stage_ids": [list(trial.declared_stage_ids) for trial in trials],
        "policy_paths": [
            sorted({path for _, path in trial.policies}) for trial in trials
        ],
        "stage_runner_version": [trial.stage_runner_version for trial in trials],
        "lerobot_version": [trial.lerobot_version for trial in trials],
        "config_version": [trial.config_version for trial in trials],
    }
    differing: dict[str, list[Any]] = {}
    for name, values in columns.items():
        # Linear dedup rather than a set: two of these are lists, which are not
        # hashable, and the trial count here is a batch of runs, not a stream.
        distinct: list[Any] = []
        for value in values:
            if value not in distinct:
                distinct.append(value)
        if len(distinct) > 1:
            differing[name] = distinct
    return differing


def summarize(
    trials: Sequence[TrialRecord],
    labels: Mapping[tuple[str, str], str],
    standalone_rates: Mapping[str, StandaloneRate] | None = None,
) -> dict[str, Any]:
    """Every metric this module knows, as one JSON-serialisable dict.

    With no labels the success metrics are None, never 0.0 and never guessed
    from terminators -- a stage that timed out cleanly is not a success, and
    printing 0.0 for "not measured" is the failure mode this whole split
    (machine events here, human outcomes in a CSV) exists to avoid. The same rule
    governs `chaining`: quantities the data cannot support carry a reason and the
    measurement that would supply them, not a number.
    """
    stage_ids = stage_order(trials)
    summary: dict[str, Any] = {
        "trial_count": len(trials),
        "completed_trial_count": sum(1 for trial in trials if trial.completed),
        "stage_ids": stage_ids,
        "timing": timing_summary(trials),
        "trials": [
            {
                "run_id": trial.run_id,
                "started_iso": trial.started_iso,
                "ended_iso": trial.ended_iso,
                "completed": trial.completed,
                "end_reason": trial.end_reason,
                "stage_count": len(trial.stages),
                # Provenance, so a quoted summary can answer "which checkpoints,
                # which fps, which robot" without going back to the JSONL.
                "robot_type": trial.robot_type,
                "fps": trial.fps,
                "dataset_repo_id": trial.dataset_repo_id,
                "declared_stage_ids": list(trial.declared_stage_ids),
                # The version columns are here as well as in `heterogeneous` so
                # a named disagreement can be traced back to the trial that
                # carries the odd value -- the heterogeneity entry lists the
                # distinct values, not who had them.
                "stage_runner_version": trial.stage_runner_version,
                "lerobot_version": trial.lerobot_version,
                "config_version": trial.config_version,
                "policies": [
                    {"stage_id": stage_id, "policy_path": policy_path}
                    for stage_id, policy_path in trial.policies
                ],
            }
            for trial in trials
        ],
        # Empty dict = checked and identical. A non-empty one does not stop the
        # aggregation, it names what differs.
        "heterogeneous": heterogeneity(trials),
    }
    if not labels:
        summary.update(
            {
                "labels_available": False,
                "scored_trial_count": 0,
                "per_stage_success_rate": None,
                "chain_success_rate": None,
                "average_sequence_length": None,
                "failure_histogram": None,
                "missing_labels": None,
                "chaining": None,
                "note": (
                    "success metrics need the labels file "
                    "(run_id,stage_id,outcome[,failure_phase,note]); "
                    "a terminator firing is not a success"
                ),
            }
        )
        return summary
    scored = scored_trials(trials, labels, stage_ids)
    scored_ids = {trial.run_id for trial in scored}
    summary.update(
        {
            "labels_available": True,
            "scored_trial_count": len(scored),
            "excluded_run_ids": [
                trial.run_id for trial in trials if trial.run_id not in scored_ids
            ],
            "per_stage_success_rate": per_stage_success_rate(trials, labels, stage_ids),
            "chain_success_rate": chain_success_rate(trials, labels, stage_ids),
            "average_sequence_length": average_sequence_length(
                trials, labels, stage_ids
            ),
            "failure_histogram": failure_histogram(trials, labels, stage_ids),
            "missing_labels": missing_label_counts(trials, labels, stage_ids),
            "chaining": chaining_metrics(
                trials, labels, stage_ids, standalone_rates=standalone_rates
            ),
            "note": (
                "the chaining loss ratio needs a STANDALONE success rate for the "
                "second stage (--standalone); every rate under `chaining` carries "
                "its own definition and denominator"
            ),
        }
    )
    return summary


def _format_optional(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_report(summary: Mapping[str, Any]) -> str:
    """The human-readable report, in this repo's `[phase]` tag style."""
    lines: list[str] = []
    timing = summary["timing"]
    lines.append(
        f"[trials] {summary['trial_count']} 회차 / runs, "
        f"{summary['completed_trial_count']} completed, "
        f"stages={summary['stage_ids']}"
    )
    for trial in summary["trials"]:
        flag = "ok" if trial["completed"] else f"INCOMPLETE ({trial['end_reason']})"
        lines.append(
            f"  - {trial['run_id']}: {trial['stage_count']} stages, "
            f"{trial['started_iso']} [{flag}]"
        )
    if summary["heterogeneous"]:
        # Named, not blocked: pooling two arms is legitimate, pooling them
        # without noticing is how a mock run's 30 Hz ends up in the same mean as
        # the real loop's ~20 Hz.
        lines.append(
            f"[warn] 이 시행들은 조건이 다르다 / pooled trials disagree on: "
            f"{summary['heterogeneous']}"
        )
    lines.append(
        "[timing] per stage: n (measured/not run), elapsed_s mean, frames mean, "
        "hertz mean/min/max -- rates over MEASURED records only"
    )
    for stage_id in summary["stage_ids"]:
        stage = timing["per_stage"][stage_id]
        lines.append(
            f"  - {stage_id}: n={stage['n']} "
            f"(measured={stage['n_measured']}, never ran={stage['not_run_n']}) "
            f"elapsed={stage['elapsed_s_mean']:.2f}s "
            f"frames={stage['frames_mean']:.1f} "
            f"hertz={stage['hertz_mean']:.2f} "
            f"({stage['hertz_min']:.2f}..{stage['hertz_max']:.2f}) "
            f"terminators(inferred)={stage['terminators']} "
            f"planned->actual={stage['planned_vs_actual']}"
        )
    gap = timing["stage_end_to_stage_start_residual_s"]
    boundary_cost = timing["boundary_cost_s"]
    stop_base = timing["stop_base_s"]
    # THE TOTAL IS PRINTED FIRST, because it is the one a reader quotes. The two
    # halves follow it named for what they are: neither contains the other since
    # the base is stopped before the stage_end emit, so a line reading "of which
    # stop_base" under the residual -- as this report said until 2026-09-08 --
    # subtracted the hardware cost out of the number instead of into it.
    lines.append(
        f"  - boundary cost (stop_base + stage_end -> stage_start residual): "
        f"n={boundary_cost['n']} of {boundary_cost['crossed_boundary_n']} crossed "
        f"mean={boundary_cost['mean']:.3f}s "
        f"({boundary_cost['min']:.3f}..{boundary_cost['max']:.3f}); "
        f"excludes the next stage's record_loop entry, so it is a LOWER BOUND"
    )
    if boundary_cost["unpaired_boundary_n"]:
        lines.append(
            f"      {boundary_cost['unpaired_boundary_n']} crossed boundaries "
            f"have no total: their stop_base fell back or failed, so its duration "
            f"is a failing call, not the cost of a stop"
        )
    lines.append(
        f"  - stage_end -> stage_start residual: n={gap['n']} "
        f"mean={gap['mean']:.3f}s ({gap['min']:.3f}..{gap['max']:.3f}); "
        f"the two emits and the bookkeeping between them -- stop_base is NOT in "
        f"here, it ran before stage_end"
    )
    lines.append(
        f"  - plus stop_base (runs before stage_end; 1 get_observation + 1 "
        f"send_action): n={stop_base['n']} mean={stop_base['mean']:.3f}s "
        f"({stop_base['min']:.3f}..{stop_base['max']:.3f})"
    )
    if timing["stop_base_fallback_n"]:
        fallback = ", ".join(
            f"{item['run_id']}/{item['stage_id']}"
            for item in timing["stop_base_fallback"]
        )
        lines.append(
            f"  - stop_base 폴백 / fell back to a direct base command at "
            f"{timing['stop_base_fallback_n']} boundaries: {fallback} -- the base "
            f"WAS stopped, but the hold action failed there (the arms were not "
            f"commanded), so those durations are excluded from the mean above"
        )
    if timing["stop_base_failed_n"]:
        failed = ", ".join(
            f"{item['run_id']}/{item['stage_id']}"
            for item in timing["stop_base_failed"]
        )
        lines.append(
            f"  - !! stop_base 실패 / FAILED at {timing['stop_base_failed_n']} "
            f"boundaries: {failed} -- the base kept the last policy velocity "
            f"until robot.disconnect(), including through save_episode when the "
            f"failure was at the last stage"
        )
    if not summary["labels_available"]:
        lines.append(
            f"[chain] 성공 지표 없음 / no success metrics -- {summary['note']}"
        )
        return "\n".join(lines)
    lines.append(
        f"[chain] scored trials={summary['scored_trial_count']} "
        f"(excluded: {summary['excluded_run_ids'] or 'none'})"
    )
    rates = summary["per_stage_success_rate"]
    for stage_id in summary["stage_ids"]:
        value = rates.get(stage_id)
        lines.append(f"  - {stage_id} success rate: {_format_optional(value)}")
    chain = summary["chain_success_rate"]
    for index, value in enumerate(chain, start=1):
        lines.append(f"  - chain_sr through stage {index}: {value:.3f}")
    lines.append(
        f"  - average sequence length: "
        f"{_format_optional(summary['average_sequence_length'])}"
    )
    lines.append(f"  - first-failure histogram: {summary['failure_histogram']}")
    missing = summary["missing_labels"]
    if any(missing.values()):
        # Printed only when there are any, and never silently: a chain_sr pulled
        # down by an untyped row must not look like one pulled down by the robot.
        lines.append(
            f"  - !! 라벨 누락 / missing label rows over scored trials: {missing} "
            f"-- these shrink a denominator, they are never counted as failures"
        )
    chaining = summary["chaining"]
    lines.append(
        f"[seam] {chaining['first_stage']} -> {chaining['second_stage']}, "
        f"{chaining['scored_trial_count']} scored trials"
    )
    for name, entry in chaining["observed"].items():
        lines.append(
            f"  - {name}: {_format_optional(entry['value'])} "
            f"({entry['numerator']}/{entry['denominator']})"
        )
        lines.append(f"      = {entry['definition']}")
    for key in ("chaining_loss_ratio", "second_stage_association_lift"):
        entry = chaining[key]
        if entry.get("available"):
            lines.append(f"  - {key}: {entry['value']:.3f}")
            lines.append(f"      = {entry['definition']}")
            # BOTH sides' counts, or the quotient is quotable with no idea how
            # many chain trials it rests on -- which is the number the seam
            # question actually turns on.
            if "numerator_source" in entry:
                lines.append(f"      numerator:   {entry['numerator_source']}")
            if "denominator_source" in entry:
                lines.append(f"      denominator: {entry['denominator_source']}")
            comparability = entry.get("comparability")
            if comparability:
                for warning in comparability["warnings"]:
                    lines.append(f"      !! 비교 조건 경고 / {warning}")
                if comparability["unchecked"]:
                    # "Checked and the same" and "could not check" must not print
                    # the same, here as everywhere else in this module.
                    lines.append(
                        f"      비교 못한 축 / not checked (the chain logs carry "
                        f"no value): {comparability['unchecked']}"
                    )
        else:
            # UNAVAILABLE is a result, and it is printed as loudly as a number
            # would be: the alternative is the reader recomputing it wrongly.
            lines.append(f"  - {key}: UNAVAILABLE")
            lines.append(f"      이유 / reason: {entry['unavailable_reason']}")
            if "measurement_required" in entry:
                lines.append(
                    f"      필요한 측정 / measurement required: "
                    f"{entry['measurement_required']}"
                )
    lines.append(f"  - {summary['note']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Module CLI. print, not logging: stdout IS the product here."""
    parser = argparse.ArgumentParser(
        prog="python -m stage_runner.aggregate",
        description=(
            "stage_runner events.jsonl 집계 / aggregate stage-chaining metrics"
        ),
    )
    parser.add_argument("events", nargs="+", type=Path, help="events.jsonl paths")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="hand-filled labels CSV: run_id,stage_id,outcome[,failure_phase,note]",
    )
    parser.add_argument(
        "--standalone",
        type=Path,
        default=None,
        help=(
            "standalone per-stage evaluation counts CSV: stage_id,trials,"
            "successes,robot_type,fps,policy_path,measured_date -- required for "
            "the chaining loss ratio, which no chain log can supply on its own; "
            "the provenance columns are required too, and the ratio is refused "
            "when they disagree with the chain trials being divided"
        ),
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the summary as JSON"
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    missing = [path for path in arguments.events if not path.is_file()]
    if missing:
        raise SystemExit(f"없는 파일 / no such file: {[str(path) for path in missing]}")
    print(f"[load] {len(arguments.events)} events.jsonl")
    trials = load_trials(arguments.events)
    labels: dict[tuple[str, str], str] = {}
    if arguments.labels is not None:
        if not arguments.labels.is_file():
            raise SystemExit(
                f"없는 labels 파일 / no such labels file: {arguments.labels}"
            )
        labels = load_labels(arguments.labels)
        print(f"[load] {len(labels)} labels from {arguments.labels}")
    standalone_rates: dict[str, StandaloneRate] = {}
    if arguments.standalone is not None:
        if not arguments.standalone.is_file():
            raise SystemExit(
                f"없는 standalone 파일 / no such standalone file: "
                f"{arguments.standalone}"
            )
        standalone_rates = load_standalone_rates(arguments.standalone)
        print(
            f"[load] {len(standalone_rates)} standalone rates from "
            f"{arguments.standalone}"
        )
    summary = summarize(trials, labels, standalone_rates)
    print(format_report(summary))
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[write] {arguments.json}")
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
