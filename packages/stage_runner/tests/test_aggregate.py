"""Unit test for stage_runner.aggregate against a frozen JSONL fixture.

Build item 18. This is the only unit test in the package, and the reason is
specific: the aggregator is the one hardware-free pure function here, it will be
re-run on event logs recorded long before whatever version is current, and when
it is wrong it does not raise -- it prints a plausible chain success rate and a
plausible loss ratio. The only other way to catch an error is to recount two
dozen trials by hand off the video sheet.

So every expected value below is a LITERAL with its arithmetic written above it,
never a value read back out of the implementation. A test that compares the code
to itself only proves the code is deterministic.

Runs either way, because pytest is not a declared dependency of this repo::

    uv run python -m pytest packages/stage_runner/tests/test_aggregate.py
    uv run python packages/stage_runner/tests/test_aggregate.py
"""

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS_DIRECTORY = Path(__file__).resolve().parent
# Running this file directly (no pytest, no installed package) puts only the
# tests directory on sys.path, so point at src/ before importing the package.
sys.path.insert(0, str(TESTS_DIRECTORY.parent / "src"))

# The E402 suppressions below stay. MEASURED both ways on this file: the PINNED
# ruff (0.12.7, .pre-commit-config.yaml) reports E402 on these two imports once
# the directive is removed, while ruff 0.16.6 no longer enables E402 by default
# and reports the directive itself as unused (RUF100). The pinned version is the
# one the commit hook runs, so it wins; whoever runs `pre-commit autoupdate` gets
# RUF100 here and `--fix` deletes these two comments for them.
from stage_runner.aggregate import (  # noqa: E402
    OUTCOME_FAIL,
    OUTCOME_SUCCESS,
    OUTCOME_VALUES,
    TRIAL_REASON_TRUNCATED,
    StageRecord,
    StandaloneRate,
    TrialRecord,
    average_sequence_length,
    build_trial,
    chain_success_rate,
    chaining_metrics,
    failure_histogram,
    format_report,
    heterogeneity,
    load_labels,
    load_standalone_rates,
    load_trials,
    missing_label_counts,
    per_stage_success_rate,
    stage_order,
    stop_base_path,
    summarize,
    timing_summary,
)
from stage_runner.results import (  # noqa: E402
    TERMINATED_BY_MANUAL,
    TERMINATED_BY_STOP_RECORDING,
    TERMINATED_BY_TIMEOUT,
    TRIAL_REASON_COMPLETED,
)

EVENTS_PATH = TESTS_DIRECTORY / "fixtures" / "events_two_trials.jsonl"
LABELS_PATH = TESTS_DIRECTORY / "fixtures" / "labels_two_trials.csv"

# The fixture, restated so a reader does not have to parse JSON to check the
# arithmetic below:
#
#   t01  s4  timeout         elapsed 12.9 s  frames 387   frame_idx 0   -> 387
#        s5  manual          elapsed  9.4 s  frames 282   frame_idx 387 -> 669
#        trial_end completed=true
#   t02  s4  timeout         elapsed 12.9 s  frames 263   frame_idx 0   -> 263
#        s5  manual          elapsed  8.6 s  frames 172   frame_idx 263 -> 435
#        trial_end completed=true
#   t03  s4  stop_recording  elapsed  3.2 s  frames  94   frame_idx 0   ->  94
#        NO trial_end (process killed) -> truncated, unlabelled
#
#   labels  t01 s4 success  t01 s5 success   t02 s4 success  t02 s5 fail
#
# THE BOUNDARY IS TWO DISJOINT HALVES, AND THE FIXTURE ENCODES THE WRITER'S
# ORDER. runner.py:181-226 runs, per boundary:
#
#   stage_start emit -> executor (elapsed_s) -> stop_base (stop_base_s)
#     -> stage_end emit -> transition emit -> next stage_start emit
#
# stop_base runs FIRST, before the stage_end emit, because a base still driving
# must not wait on bookkeeping. So every stage_end stamp below sits at
#
#   stage_start + elapsed_s + stop_base_s + 0.010
#
# where 0.010 s is the part of the stage that is neither the executor's
# perf_counter window nor stop_base (record_loop entry/exit outside the window,
# plus the emit itself), and the stage_end -> stage_start RESIDUAL that follows
# contains neither of them:
#
#   t01  s4 stage_start 14:12:05.004  + 12.900 + 0.031 + 0.010 = end 17.945
#        transition     14:12:17.947 (end + 0.002)   stop_base_s 0.031
#        s5 stage_start 14:12:17.948 (tr  + 0.001)
#          -> residual 17.948 - 17.945 = 0.003 s ; boundary cost 0.031 + 0.003
#             = 0.034 s
#   t02  s4 stage_start 15:03:12.771  + 12.900 + 0.028 + 0.010 = end 25.709
#        transition     15:03:25.710 (end + 0.001)   stop_base_s 0.028
#        s5 stage_start 15:03:25.711 (tr  + 0.001)
#          -> residual 25.711 - 25.709 = 0.002 s ; boundary cost 0.028 + 0.002
#             = 0.030 s
#   t03  one stage_end, no stage_start after it -> no residual, no cost
#
# THE RESIDUAL IS AN ORDER OF MAGNITUDE SMALLER THAN stop_base HERE, and that is
# the point: a 0.031 s base stop shows up in it as 0.003 s, a 0.300 s one would
# show up as 0.003 s too. Whoever reads the residual as "the transition gap"
# reads the hardware cost as zero. The old fixture encoded the old order (gap
# 0.033 s > stop_base 0.031 s, so the gap looked like it contained the call).
#
# A TRANSITION LINE PER STAGE, NOT PER CROSSING. The runner emits one after
# EVERY stage -- after the last one and after an abort too, with to_stage_id
# null -- because a stop_base that silently failed must not produce a log
# byte-identical to a clean run. So the fixture carries five of them for five
# stages, only two of which are crossings a residual can be measured across, and
# all THREE stop_base paths appear:
#
#   t01  s4 -> s5   primary  0.031      t01  s5 -> null  primary  0.029
#   t02  s4 -> s5   primary  0.028      t02  s5 -> null  FALLBACK 0.734
#   t03  s4 -> null FAILED   0.512  (the aborted stage; stop_action null)
#
# Only the primary path is a timing sample. On the other two, stop_base_s is how
# long the failing get_observation took before it raised, not what stopping the
# base costs -- and the two are not the same event either: the FALLBACK zeroed
# the base directly (stopped, primary mechanism broken), the FAILED one left it
# possibly still driving. stop_action is null on BOTH, which is exactly why the
# reader keys off stop_base_path instead. Those durations still elapsed before
# their stage_end (the call was made either way), so the stamps above include
# them -- but they are never summed into a boundary cost, because a failing
# camera read is not what crossing the seam costs.
#
# PROVENANCE ON BOTH SIDES OF THE JOIN. The fixture's trial_start lines carry
# robot_type "mobileai_robot", fps 30 and the two policy paths below, and the
# standalone rows in these tests carry the same -- because the loss ratio joins
# them on stage_id alone and then divides one into the other.
TOLERANCE = 1e-9

# The fixture's own provenance, restated so the standalone rows below can be
# read as matching it without opening the JSONL.
FIXTURE_ROBOT_TYPE = "mobileai_robot"
FIXTURE_FPS = 30
FIXTURE_S5_POLICY = "kiroaiseoul/act_task05_tube_disposal_60000"
# make_trial()'s pool, which is a different (hand-built) one.
MOCK_S5_POLICY = "mock://five"


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=TOLERANCE)


def standalone(
    trials: int,
    successes: int,
    *,
    stage_id: str = "s5",
    robot_type: str = FIXTURE_ROBOT_TYPE,
    fps: int = FIXTURE_FPS,
    policy_path: str = MOCK_S5_POLICY,
    measured_date: str = "2026-09-10",
) -> StandaloneRate:
    """A standalone row that MATCHES the pool it will be divided into.

    The defaults are make_trial()'s conditions, because that is the pool most of
    these tests build. Provenance is not optional on StandaloneRate: it is joined
    to the chain on stage_id alone and then becomes a denominator, so a rate from
    another robot, fps or checkpoint would divide in unnoticed. Tests that want
    the mismatch pass it explicitly, which is what
    test_standalone_from_another_condition_is_refused does.
    """
    return StandaloneRate(
        stage_id=stage_id,
        trials=trials,
        successes=successes,
        robot_type=robot_type,
        fps=fps,
        policy_path=policy_path,
        measured_date=measured_date,
    )


def load_fixture() -> tuple[list[TrialRecord], dict[tuple[str, str], str]]:
    return load_trials([EVENTS_PATH]), load_labels(LABELS_PATH)


def test_build_trial_reads_two_stages() -> None:
    trials, _ = load_fixture()
    assert len(trials) == 3, f"세 트라이얼 / three trials, got {len(trials)}"
    trial = trials[0]
    assert trial.run_id == "t01", trial.run_id
    assert trial.completed is True, "t01은 완료됨 / t01 completed"
    assert trial.end_reason == TRIAL_REASON_COMPLETED, trial.end_reason
    assert len(trial.stages) == 2, f"두 스테이지 / two stages, got {len(trial.stages)}"

    first, second = trial.stages
    assert first.stage_id == "s4" and first.stage_index == 0, first
    assert first.planned_terminator == TERMINATED_BY_TIMEOUT, first.planned_terminator
    assert first.terminated_by == TERMINATED_BY_TIMEOUT, first.terminated_by
    assert first.frames == 387 and close(first.elapsed_s, 12.9), first
    # frame_idx comes from episode_buffer["size"], so stage 2 starts where
    # stage 1 ended: 0 -> 387 -> 669. One continuous episode, not two.
    assert first.start_frame_idx == 0 and first.end_frame_idx == 387, first
    assert second.start_frame_idx == 387 and second.end_frame_idx == 669, second
    assert second.stage_id == "s5" and second.stage_index == 1, second
    assert second.terminated_by == TERMINATED_BY_MANUAL, second.terminated_by
    assert first.start_wall_clock_iso == "2026-09-08T14:12:05.004+09:00", first
    # N stages -> N boundaries but only N-1 RESIDUALS. stop_base runs after every
    # stage including the last, and since 2026-09-08 every one of those calls is
    # logged (the last with to_stage_id null), so t01 contributes two stop_base
    # durations. A residual needs two logged ends, so the final boundary -- which
    # no stage_start follows -- contributes none.
    #
    # The residual is DERIVED from the two wall clocks (s4 stage_end
    # 14:12:17.945 -> s5 stage_start 14:12:17.948 = 0.003 s), not read from a
    # field. It is the SECOND half of the boundary: stop_base ran before the
    # stage_end emit, so its 0.031 s is not in there and the total is the sum,
    # 0.031 + 0.003 = 0.034 s.
    assert trial.transition_gaps_s == (0.003,), trial.transition_gaps_s
    assert trial.stop_base_durations_s == (0.031, 0.029), trial.stop_base_durations_s
    assert trial.stop_base_fallback_stage_ids == (), trial
    assert trial.stop_base_failed_stage_ids == (), trial.stop_base_failed_stage_ids
    # THE RELATION IS THE OTHER WAY ROUND NOW, and this assertion is the pin.
    # While stop_base ran between the two emits the gap contained it and was
    # necessarily larger; since the safety fix moved the call BEFORE the
    # stage_end emit the residual is the leftover bookkeeping and is smaller by
    # an order of magnitude. Reading it as the transition gap reports a 0.031 s
    # base stop as 0.003 s of dead time -- and a 0.300 s one as 0.003 s as well.
    assert trial.transition_gaps_s[0] < trial.stop_base_durations_s[0], (
        "잔여는 stop_base를 포함하지 않는다 / the residual does NOT contain the "
        "stop_base call: it runs before the stage_end emit"
    )
    # The two halves added per boundary -- the only one of the three numbers that
    # answers what crossing the seam cost. One entry, not two: t01's final
    # boundary crossed nothing, so it has no residual to add to.
    assert close(trial.boundary_costs_s[0], 0.034), trial.boundary_costs_s
    assert len(trial.boundary_costs_s) == 1, trial.boundary_costs_s
    assert close(
        trial.boundary_costs_s[0],
        trial.stop_base_durations_s[0] + trial.transition_gaps_s[0],
    ), trial.boundary_costs_s
    # Provenance off trial_start, so a pooled summary can say what produced it
    # instead of averaging a mock run into the same hertz_mean as the real loop.
    assert trial.robot_type == "mobileai_robot", trial.robot_type
    assert trial.fps == 30, trial.fps
    assert trial.dataset_repo_id == "kiroaiseoul/eval_chain_task45_t01", trial
    assert trial.declared_stage_ids == ("s4", "s5"), trial.declared_stage_ids
    assert trial.policies == (
        ("s4", "kiroaiseoul/act_task04_pour_liquid_from_tubes_to_beaker_60000"),
        ("s5", "kiroaiseoul/act_task05_tube_disposal_60000"),
    ), trial.policies
    # The three code versions, the axis the four fields above cannot see: two
    # trials can agree on robot, fps and checkpoints and still have been measured
    # by different software. Parsed here so heterogeneity() can compare them.
    assert trial.stage_runner_version == "0.1.0", trial.stage_runner_version
    assert trial.lerobot_version == "0.4.4", trial.lerobot_version
    assert trial.config_version == 1, trial.config_version
    assert stage_order(trials) == ["s4", "s5"], stage_order(trials)


def test_hertz_is_frames_over_elapsed() -> None:
    trials, _ = load_fixture()
    t01_s4, t02_s4 = trials[0].stages[0], trials[1].stages[0]
    # 387 frames / 12.9 s = 30.0 Hz exactly -- the 30 fps the YAML timers assume.
    assert close(t01_s4.hertz, 30.0), t01_s4.hertz
    # 263 / 12.9 = 2630/129 = 20.3875968992248062... -- the slow-loop case another
    # session measured. Same timeout, ~2/3 of the frames, so the same action
    # sequence does NOT fit in the same wall clock.
    assert close(t02_s4.hertz, 20.387596899224807), t02_s4.hertz
    # elapsed_s <= 0 is a division by zero, and 0.0 is the only answer that is
    # not a lie about the rate.
    zero = StageRecord(
        stage_id="s4",
        stage_index=0,
        planned_terminator=TERMINATED_BY_TIMEOUT,
        terminated_by=TERMINATED_BY_TIMEOUT,
        reason="",
        elapsed_s=0.0,
        frames=0,
        start_frame_idx=0,
        end_frame_idx=0,
        start_wall_clock_iso="",
        end_wall_clock_iso="",
    )
    assert zero.hertz == 0.0, zero.hertz


def test_timing_summary_includes_the_truncated_trial() -> None:
    trials, _ = load_fixture()
    timing = timing_summary(trials)
    assert timing["trial_count"] == 3, timing["trial_count"]
    assert timing["completed_trial_count"] == 2, timing["completed_trial_count"]

    # s4 ran in all three trials, INCLUDING the truncated one: a stage that
    # finished is a real measurement even when the trial after it did not.
    s4 = timing["per_stage"]["s4"]
    assert s4["n"] == 3, s4["n"]
    # (12.9 + 12.9 + 3.2) / 3 = 29.0 / 3 = 9.66666666...
    assert close(s4["elapsed_s_mean"], 9.666666666666666), s4["elapsed_s_mean"]
    # (387 + 263 + 94) / 3 = 744 / 3 = 248
    assert close(s4["frames_mean"], 248.0), s4["frames_mean"]
    # (30.0 + 20.387596899224807 + 29.375) / 3 = 79.76259689922481 / 3
    assert close(s4["hertz_mean"], 26.5875322997416), s4["hertz_mean"]
    assert close(s4["hertz_min"], 20.387596899224807), s4["hertz_min"]
    assert close(s4["hertz_max"], 30.0), s4["hertz_max"]
    # The operator abort stays visible here rather than hiding inside the mean.
    assert s4["terminators"] == {
        TERMINATED_BY_TIMEOUT: 2,
        TERMINATED_BY_STOP_RECORDING: 1,
    }, s4["terminators"]

    # Every fixture stage actually ran, so nothing is excluded here; the
    # exclusion itself is tested in test_stage_that_never_ran_is_not_a_slow_stage.
    assert s4["n_measured"] == 3 and s4["not_run_n"] == 0, s4
    # The terminator counts are INFERRED (record_loop self-clears exit_early), so
    # the planned->actual pairs are carried beside them: two stages planned as a
    # timeout ended as one, and the third was ended by the operator.
    assert s4["planned_vs_actual"] == {
        "timeout->timeout": 2,
        "timeout->stop_recording": 1,
    }, s4["planned_vs_actual"]

    s5 = timing["per_stage"]["s5"]
    assert s5["n"] == 2, s5["n"]
    # (9.4 + 8.6) / 2 = 18.0 / 2 = 9.0 ; (30.0 + 20.0) / 2 = 25.0
    assert close(s5["elapsed_s_mean"], 9.0), s5["elapsed_s_mean"]
    assert close(s5["hertz_mean"], 25.0), s5["hertz_mean"]
    assert s5["terminators"] == {TERMINATED_BY_MANUAL: 2}, s5["terminators"]

    # The stage_end -> stage_start RESIDUAL, derived per trial: t01 0.003 s,
    # t02 0.002 s. (0.003 + 0.002) / 2 = 0.0025. t03 has no second stage, so it
    # contributes nothing to n.
    gap = timing["stage_end_to_stage_start_residual_s"]
    assert gap["n"] == 2, gap["n"]
    assert close(gap["mean"], 0.0025), gap["mean"]
    assert close(gap["min"], 0.002) and close(gap["max"], 0.003), gap
    # The caveat travels in the machine-readable output too, not only in the
    # report string a JSON reader never sees -- and stop_base is now the FIRST
    # thing it excludes, since a reader who misses that reads a 0.300 s base stop
    # as 0.000 s of transition.
    assert "record_loop entry" in gap["excludes"], gap
    assert gap["excludes"].startswith("stop_base"), gap["excludes"]
    assert "BEFORE the stage_end emit" in gap["excludes"], gap["excludes"]
    # The stop_base call alone, reported separately and NEVER as the gap. Five
    # boundaries are logged (stop_base runs after the last stage too) but only
    # the THREE primary-path ones are timing samples: t02's final boundary fell
    # back to a direct base command and t03's failed outright, and on both of
    # those stop_base_s is a failing camera read, not the cost of a stop.
    # (0.031 + 0.029 + 0.028) / 3 = 0.088 / 3 = 0.0293333333333333...
    stop_base = timing["stop_base_s"]
    assert stop_base["n"] == 3, stop_base["n"]
    assert close(stop_base["mean"], 0.029333333333333333), stop_base["mean"]
    assert close(stop_base["min"], 0.028) and close(stop_base["max"], 0.031), stop_base
    # FLIPPED with the safety fix, and this is the whole measurement point: the
    # base is stopped before the stage_end emit, so the residual left after it is
    # an order of magnitude SMALLER than the call it no longer contains. Whoever
    # quotes the residual as the transition gap quotes 0.0025 s for a boundary
    # that cost 0.032 s.
    assert stop_base["mean"] > gap["mean"], (stop_base, gap)
    # THE TOTAL, paired per boundary: t01 0.031 + 0.003 = 0.034, t02 0.028 +
    # 0.002 = 0.030. (0.034 + 0.030) / 2 = 0.064 / 2 = 0.032. Two of the five
    # boundaries are in it -- the other three crossed nothing.
    boundary_cost = timing["boundary_cost_s"]
    assert boundary_cost["n"] == 2, boundary_cost["n"]
    assert close(boundary_cost["mean"], 0.032), boundary_cost["mean"]
    assert close(boundary_cost["min"], 0.030) and close(boundary_cost["max"], 0.034), (
        boundary_cost
    )
    # Both crossed boundaries had a primary stop_base, so nothing is unpaired
    # here; the unpaired case is asserted in
    # test_a_non_primary_stop_base_has_no_boundary_total.
    assert boundary_cost["crossed_boundary_n"] == 2, boundary_cost
    assert boundary_cost["unpaired_boundary_n"] == 0, boundary_cost
    # The sum is NOT the two reported means added: those are over different
    # samples (n=2 and n=3). It happens to agree here only because both crossed
    # boundaries took the primary path -- 0.032 vs 0.0025 + 0.029333... =
    # 0.031833..., which is already a different number.
    assert not close(boundary_cost["mean"], gap["mean"] + stop_base["mean"]), (
        boundary_cost,
        gap,
        stop_base,
    )
    # The fallback is NOT a failure -- the base was stopped -- but it says the
    # hold action broke, which is invisible in the mean above by construction.
    assert timing["stop_base_fallback_n"] == 1, timing["stop_base_fallback_n"]
    assert timing["stop_base_fallback"] == [{"run_id": "t02", "stage_id": "s5"}], (
        timing["stop_base_fallback"]
    )
    # A base that nothing stopped is the one thing that must not read as a clean
    # run. It is counted and identified, run and stage.
    assert timing["stop_base_failed_n"] == 1, timing["stop_base_failed_n"]
    assert timing["stop_base_failed"] == [{"run_id": "t03", "stage_id": "s4"}], timing[
        "stop_base_failed"
    ]
    # 0.734 s (fallback) and 0.512 s (failed) are both far above every primary
    # sample, so pooling them would have moved the mean by an order of magnitude
    # while looking like an ordinary slow boundary.
    assert stop_base["max"] < 0.5, stop_base


def test_per_stage_and_chain_success_rate() -> None:
    trials, labels = load_fixture()
    stage_ids = stage_order(trials)
    # Scored trials are t01 and t02. t03 is dropped twice over: not completed,
    # and unlabelled. An unlabelled trial is missing data, NOT a failed one --
    # counting it as a failure would make every rate below 2/3 instead of 2/2.
    # s4: t01 success, t02 success                       -> 2/2 = 1.0
    # s5: t01 success, t02 fail                          -> 1/2 = 0.5
    rates = per_stage_success_rate(trials, labels, stage_ids)
    assert rates == {"s4": 1.0, "s5": 0.5}, rates
    # chain_sr[i] = P(stages 1..i all succeeded), over the 2 scored trials:
    #   through s4:       t01 yes, t02 yes                -> 2/2 = 1.0
    #   through s4 AND s5: t01 yes, t02 no                -> 1/2 = 0.5
    chain = chain_success_rate(trials, labels, stage_ids)
    assert chain == [1.0, 0.5], chain


def test_average_sequence_length() -> None:
    trials, labels = load_fixture()
    stage_ids = stage_order(trials)
    # leading consecutive successes: t01 = 2 (s4, s5), t02 = 1 (s4 only)
    # (2 + 1) / 2 = 1.5
    average = average_sequence_length(trials, labels, stage_ids)
    assert close(average, 1.5), average
    # Identity worth pinning: the mean chain length is the sum of the chain_sr
    # vector. 1.0 + 0.5 = 1.5. If one of the two ever drifts this breaks.
    assert close(average, sum(chain_success_rate(trials, labels, stage_ids))), average


def test_failure_histogram_counts_first_failure() -> None:
    trials, labels = load_fixture()
    stage_ids = stage_order(trials)
    # t01 never fails -> contributes to no bucket.
    # t02 first fails at s5 -> s5 += 1.
    # t03 is unscored (truncated, unlabelled) -> contributes to no bucket. Its
    # stop_recording terminator is NOT a failure: the operator ended the run,
    # which says nothing about whether the stage was working.
    histogram = failure_histogram(trials, labels, stage_ids)
    assert histogram == {"s4": 0, "s5": 1}, histogram


def test_chaining_metrics_on_the_fixture() -> None:
    """What the chain CAN say about the seam, and what it refuses to say.

    Over the 2 scored trials (t01, t02; t03 is truncated and unlabelled):
        P(s4)                  = 2/2 = 1.0
        P(s5 in chain)         = 1/2 = 0.5   labelled in both, succeeded in t01
        P(s5 | s4 succeeded)   = 1/2 = 0.5   both entered s5 out of a good s4
        P(s5 | s4 failed)      = 0/0 -> None s4 never failed here
        P(s4 and s5)           = 1/2 = 0.5
    Every one of those is a count over trials that ran. The LOSS RATIO is not:
    its denominator is s5's STANDALONE rate, which no chain trial can produce,
    so it is UNAVAILABLE until one is supplied -- and the reason has to name the
    substitution it is refusing, because the previous build made exactly that
    substitution and printed 1.000 and 1.000 off this fixture.
    """
    trials, labels = load_fixture()
    stage_ids = stage_order(trials)
    metrics = chaining_metrics(trials, labels, stage_ids)
    assert metrics["first_stage"] == "s4", metrics["first_stage"]
    assert metrics["second_stage"] == "s5", metrics["second_stage"]
    assert metrics["scored_trial_count"] == 2, metrics["scored_trial_count"]

    observed = metrics["observed"]
    first = observed["first_stage_success_rate"]
    assert close(first["value"], 1.0) and (
        first["numerator"],
        first["denominator"],
    ) == (
        2,
        2,
    ), first
    in_chain = observed["second_stage_success_rate_in_chain"]
    assert close(in_chain["value"], 0.5) and (
        in_chain["numerator"],
        in_chain["denominator"],
    ) == (1, 2), in_chain
    conditional = observed["second_stage_success_rate_given_first_success"]
    assert close(conditional["value"], 0.5) and (
        conditional["numerator"],
        conditional["denominator"],
    ) == (1, 2), conditional
    # No trial had s4 fail, so this one has an EMPTY denominator and reports
    # None. 0.0 would say "s5 never worked after a failed s4", which is a claim
    # about a case the fixture does not contain.
    after_failure = observed["second_stage_success_rate_given_first_failure"]
    assert after_failure["value"] is None, after_failure
    assert after_failure["denominator"] == 0, after_failure
    joint = observed["joint_success_rate"]
    assert close(joint["value"], 0.5) and (
        joint["numerator"],
        joint["denominator"],
    ) == (1, 2), joint
    # The definition ships WITH each number, because these five differ only in
    # which trials are in the denominator.
    for name, entry in observed.items():
        assert entry["definition"].strip(), name
    assert "NOT a standalone rate" in in_chain["definition"], in_chain["definition"]

    ratio = metrics["chaining_loss_ratio"]
    assert ratio["available"] is False and ratio["value"] is None, ratio
    # The reason has to name the number that is missing AND the artifact the
    # substitution would produce, or the next reader makes the substitution.
    assert "standalone" in ratio["unavailable_reason"], ratio["unavailable_reason"]
    assert "1/P(s4)" in ratio["unavailable_reason"], ratio["unavailable_reason"]
    assert "ALONE" in ratio["measurement_required"], ratio["measurement_required"]

    # The association lift is refused here too, and for an arithmetic reason: s4
    # never failed, so P(A)=1.0 and the lift is 1.000 whatever the seam does.
    lift = metrics["second_stage_association_lift"]
    assert lift["available"] is False and lift["value"] is None, lift
    assert "1.000" in lift["unavailable_reason"], lift["unavailable_reason"]

    # SUPPLY THE MISSING MEASUREMENT and the ratio appears, once:
    #   standalone s5 = 8 successes in 10 trials run alone      = 0.8
    #   P(s5 | s4 succeeded) = 1/2                              = 0.5
    #   loss ratio = 0.5 / 0.8                                  = 0.625
    # Below 1.0, i.e. s5 keeps 62.5% of its standalone capability when it is
    # entered from s4's terminal state instead of a hand-set one.
    with_standalone = chaining_metrics(
        trials,
        labels,
        stage_ids,
        standalone_rates={"s5": standalone(10, 8, policy_path=FIXTURE_S5_POLICY)},
    )
    supplied = with_standalone["chaining_loss_ratio"]
    assert supplied["available"] is True, supplied
    assert close(supplied["value"], 0.625), supplied["value"]
    assert close(supplied["denominator_value"], 0.8), supplied
    assert "8/10" in supplied["denominator_source"], supplied["denominator_source"]
    # BOTH SIDES' COUNTS TRAVEL, not just the denominator's. 0.625 off 1 chain
    # trial and 0.625 off 40 are not the same claim, and the seam question turns
    # on which it is -- here it rests on 1 success in 2 chain trials, which the
    # entry has to say out loud since the value itself cannot.
    assert supplied["numerator_successes"] == 1, supplied
    assert supplied["numerator_trials"] == 2, supplied
    assert close(supplied["numerator_value"], 0.5), supplied
    assert "1/2" in supplied["numerator_source"], supplied["numerator_source"]
    assert "s4" in supplied["numerator_source"], supplied["numerator_source"]
    assert supplied["denominator_successes"] == 8, supplied
    assert supplied["denominator_trials"] == 10, supplied
    # And the provenance it was checked against is on the available entry too, so
    # the reader can see WHY this denominator belongs to this numerator.
    comparability = supplied["comparability"]
    assert comparability["comparable"] is True, comparability
    assert comparability["mismatches"] == [], comparability
    assert comparability["checked"]["fps"] == {
        "standalone": 30,
        "chain": [30],
    }, comparability["checked"]
    # 2026-09-10 standalone vs 2026-09-08 chain trials = 2 days, inside the
    # review trigger, so no warning.
    assert comparability["checked"]["measured_date"]["max_separation_days"] == 2, (
        comparability["checked"]
    )
    assert comparability["warnings"] == [], comparability["warnings"]

    # The printed line carries both counts too, since that is where the number
    # actually gets quoted from.
    printed = format_report(
        summarize(
            trials,
            labels,
            {"s5": standalone(10, 8, policy_path=FIXTURE_S5_POLICY)},
        )
    )
    assert "chaining_loss_ratio: 0.625" in printed, printed
    assert "numerator:   in-chain: 1/2" in printed, printed
    assert "8/10 trials run alone" in printed, printed

    # And the report says UNAVAILABLE out loud rather than omitting the line: a
    # missing metric that prints nothing reads as a metric that was fine.
    report = format_report(summarize(trials, labels))
    assert "chaining_loss_ratio: UNAVAILABLE" in report, report
    assert "measurement required" in report, report


def test_chaining_metrics_return_one_on_independent_stages() -> None:
    """The property the previous build failed: independence must print 1.0.

    Four hand-built trials in which the two stages are independent BY
    CONSTRUCTION -- every combination appears exactly once, so P(A and B) is
    exactly P(A)P(B):

        x1  A ok    B ok        x2  A ok    B fail
        x3  A fail  B ok        x4  A fail  B fail

        P(A) = 2/4 = 0.5    P(B in chain) = 2/4 = 0.5
        P(A and B) = 1/4 = 0.25 = 0.5 * 0.5           <- independent
        P(B | A ok)   = 1/2 = 0.5
        P(B | A fail) = 1/2 = 0.5                     <- and visibly so

        association lift = P(A and B) / (P(A) * P(B)) = 0.25 / 0.25 = 1.0
        loss ratio (standalone P(B) = 2/4 = 0.5)      = 0.5 / 0.5   = 1.0

    THE PREVIOUS BUILD PRINTED 2.0 HERE for its "conditional" variant, because
    that variant divided P(A) out of the numerator and left it in the
    denominator, making it identically unconditional/P(A). Its own docstring
    said 1.0 meant independence, so on perfectly independent stages the
    instrument reported that chaining beat independence by 2x -- and the error
    factor grew as stage A got worse (4x at P(A)=0.25). This test is that
    property, pinned.
    """
    trials = [make_trial("x1"), make_trial("x2"), make_trial("x3"), make_trial("x4")]
    labels = {
        ("x1", "s4"): OUTCOME_SUCCESS,
        ("x1", "s5"): OUTCOME_SUCCESS,
        ("x2", "s4"): OUTCOME_SUCCESS,
        ("x2", "s5"): OUTCOME_FAIL,
        ("x3", "s4"): OUTCOME_FAIL,
        ("x3", "s5"): OUTCOME_SUCCESS,
        ("x4", "s4"): OUTCOME_FAIL,
        ("x4", "s5"): OUTCOME_FAIL,
    }
    stage_ids = ["s4", "s5"]
    metrics = chaining_metrics(
        trials,
        labels,
        stage_ids,
        standalone_rates={"s5": standalone(4, 2)},
    )
    observed = metrics["observed"]
    assert close(observed["first_stage_success_rate"]["value"], 0.5)
    assert close(observed["second_stage_success_rate_in_chain"]["value"], 0.5)
    assert close(
        observed["second_stage_success_rate_given_first_success"]["value"], 0.5
    )
    assert close(
        observed["second_stage_success_rate_given_first_failure"]["value"], 0.5
    )
    assert close(observed["joint_success_rate"]["value"], 0.25)

    lift = metrics["second_stage_association_lift"]
    assert lift["available"] is True, lift
    assert close(lift["value"], 1.0), lift["value"]
    assert lift["cells"] == {"first_ok_second_ok": 1, "first_failed_second_ok": 1}, lift

    ratio = metrics["chaining_loss_ratio"]
    assert ratio["available"] is True, ratio
    assert close(ratio["value"], 1.0), ratio["value"]

    # The rest of the metric family on the same four trials, unchanged:
    # chain_sr = [2/4, 1/4]; mean chain length = (2 + 1 + 0 + 0) / 4 = 0.75;
    # first failures: x3 and x4 at s4, x2 at s5.
    assert chain_success_rate(trials, labels, stage_ids) == [0.5, 0.25]
    assert close(average_sequence_length(trials, labels, stage_ids), 0.75)
    assert failure_histogram(trials, labels, stage_ids) == {"s4": 2, "s5": 1}
    # A stage nothing succeeded at leaves the ratio undefined, not zero and not
    # infinite -- and the reason says which stage.
    all_failed = {key: OUTCOME_FAIL for key in labels}
    failed_metrics = chaining_metrics(
        trials,
        all_failed,
        stage_ids,
        standalone_rates={"s5": standalone(4, 2)},
    )
    assert failed_metrics["chaining_loss_ratio"]["value"] is None, failed_metrics
    assert failed_metrics["second_stage_association_lift"]["value"] is None


def test_in_chain_rate_is_never_used_as_the_standalone_denominator() -> None:
    """The normal chain shape, where the old denominator inverted the answer.

    Ten completed, labelled trials of the ordinary kind: a failed pour leaves
    nothing to dispose of, so stage B only ever succeeds inside a trial where
    stage A succeeded.

        A succeeded in 5 of 10                     P(A)            = 0.5
        B succeeded in 2 of those 5                P(B | A ok)     = 0.4
        B succeeded in 0 of the 5 A-failures       P(B | A fail)   = 0.0
        B in chain: 2 of 10                        P(B in chain)   = 0.2
        joint: 2 of 10                             P(A and B)      = 0.2

    B's STANDALONE rate is 8/10 = 0.8 (measured separately, from a hand-set
    start state), so the true seam ratio is 0.4 / 0.8 = 0.5: the chain keeps
    half of B's capability. The previous build printed 2.0 and 4.0 -- both above
    1.0, i.e. "chaining helps", both wrong in the opposite direction from the
    truth, and both equal to 1/P(A) and 1/P(A)^2 rather than to anything about
    the seam. The reason is arithmetic: with every B success inside an A
    success, P(A and B) == P(B in chain), so P(A and B)/(P(A)*P(B in chain))
    cancels to 1/P(A) exactly.

    So the lift is reported UNAVAILABLE here with that identity named, and the
    ratio is computed only against the standalone number.
    """
    trials = [make_trial(f"n{index}") for index in range(1, 11)]
    labels: dict[tuple[str, str], str] = {}
    for index in range(1, 11):
        run_id = f"n{index}"
        first_ok = index <= 5
        second_ok = index <= 2
        labels[(run_id, "s4")] = OUTCOME_SUCCESS if first_ok else OUTCOME_FAIL
        labels[(run_id, "s5")] = OUTCOME_SUCCESS if second_ok else OUTCOME_FAIL
    stage_ids = ["s4", "s5"]
    metrics = chaining_metrics(
        trials,
        labels,
        stage_ids,
        standalone_rates={"s5": standalone(10, 8)},
    )
    observed = metrics["observed"]
    assert close(observed["first_stage_success_rate"]["value"], 0.5)
    assert close(observed["second_stage_success_rate_in_chain"]["value"], 0.2)
    assert close(
        observed["second_stage_success_rate_given_first_success"]["value"], 0.4
    )
    assert close(
        observed["second_stage_success_rate_given_first_failure"]["value"], 0.0
    )
    assert close(observed["joint_success_rate"]["value"], 0.2)

    lift = metrics["second_stage_association_lift"]
    assert lift["available"] is False, lift
    # 1/P(A) = 1/0.5 = 2.000, and that is the whole content of the number.
    assert "1/P(s4) = 2.000" in lift["unavailable_reason"], lift["unavailable_reason"]

    # 0.4 / 0.8 = 0.5 -- the chain keeps half of s5's standalone capability.
    ratio = metrics["chaining_loss_ratio"]
    assert ratio["available"] is True, ratio
    assert close(ratio["value"], 0.5), ratio["value"]


def test_missing_label_shrinks_one_denominator_and_is_counted() -> None:
    """One untyped row must not make one stage report three different rates.

    Four completed trials, s4 succeeded in all four, s5 succeeded in 2 of the 3
    rows the operator typed and the fourth s5 row was never typed (a gap off the
    paper sheet).

        per-stage s5 success rate : 2/3 = 0.667   denominator = LABELLED trials
        chain_sr through s5       : 2/4 = 0.5     a missing label breaks a chain
        P(s5 in chain)            : 2/3 = 0.667   SAME rule as the per-stage rate

    The third of those used to be 2/4 = 0.5, because loss_ratio divided by the
    trial count and so counted an untyped row as a stage-B failure -- the one
    place in this module that broke its own asymmetry rule ("a missing label is
    not a failure either"). The two rates that answer the same question now
    agree, and the ones that legitimately differ are accompanied by the count
    that explains the difference.
    """
    trials = [make_trial(f"m{index}") for index in range(1, 5)]
    labels = {
        ("m1", "s4"): OUTCOME_SUCCESS,
        ("m1", "s5"): OUTCOME_SUCCESS,
        ("m2", "s4"): OUTCOME_SUCCESS,
        ("m2", "s5"): OUTCOME_SUCCESS,
        ("m3", "s4"): OUTCOME_SUCCESS,
        ("m3", "s5"): OUTCOME_FAIL,
        ("m4", "s4"): OUTCOME_SUCCESS,
        # m4 s5: not typed.
    }
    stage_ids = ["s4", "s5"]
    two_thirds = 2.0 / 3.0
    rates = per_stage_success_rate(trials, labels, stage_ids)
    assert close(rates["s5"], two_thirds), rates
    assert chain_success_rate(trials, labels, stage_ids) == [1.0, 0.5]
    in_chain = chaining_metrics(trials, labels, stage_ids)["observed"][
        "second_stage_success_rate_in_chain"
    ]
    assert close(in_chain["value"], two_thirds), in_chain
    assert in_chain["denominator"] == 3, in_chain
    assert close(in_chain["value"], rates["s5"]), (in_chain, rates)
    # And the count is reported, so a chain_sr depressed by data entry is
    # distinguishable from one depressed by the robot.
    assert missing_label_counts(trials, labels, stage_ids) == {"s4": 0, "s5": 1}
    summary = summarize(trials, labels)
    assert summary["missing_labels"] == {"s4": 0, "s5": 1}, summary["missing_labels"]
    assert "missing label rows" in format_report(summary)


def test_stage_that_never_ran_is_not_a_slow_stage() -> None:
    """The per-stage Hz the first-run timer procedure is read off.

    Four trials of stage s5: three ran 270 frames in 9.0 s (30.000 Hz exactly)
    and in the fourth the operator pressed Esc during the s4 -> s5 gap, so s5
    never entered record_loop -- executors._aborted_before_start returns
    elapsed_s=0.0, frames=0 and the runner emits a full stage_end for it.

        pooled (the old behaviour):  (30 + 30 + 30 + 0) / 4 = 22.500 Hz, min 0.000
        measured only:               (30 + 30 + 30) / 3     = 30.000 Hz, min 30.000

    The operator following the MANDATORY FIRST-RUN PROCEDURE reads this number
    out of events.jsonl and sets every timeout in the YAML from it, so the
    pooled version lengthens the timers by 25% on the strength of a stage that
    recorded nothing. The record stays visible in n, not_run_n and the
    terminator histogram -- excluded from the rate, not hidden.
    """
    trials = [
        make_single_stage_trial("z1", elapsed_s=9.0, frames=270),
        make_single_stage_trial("z2", elapsed_s=9.0, frames=270),
        make_single_stage_trial("z3", elapsed_s=9.0, frames=270),
        make_single_stage_trial(
            "z4",
            elapsed_s=0.0,
            frames=0,
            terminated_by=TERMINATED_BY_STOP_RECORDING,
            planned_terminator=TERMINATED_BY_MANUAL,
        ),
    ]
    stage = timing_summary(trials)["per_stage"]["s5"]
    assert stage["n"] == 4, stage["n"]
    assert stage["n_measured"] == 3, stage["n_measured"]
    assert stage["not_run_n"] == 1, stage["not_run_n"]
    assert close(stage["hertz_mean"], 30.0), stage["hertz_mean"]
    assert close(stage["hertz_min"], 30.0), stage["hertz_min"]
    assert close(stage["frames_mean"], 270.0), stage["frames_mean"]
    assert close(stage["elapsed_s_mean"], 9.0), stage["elapsed_s_mean"]
    # Still counted, still visible, still attributed to the operator.
    assert stage["terminators"] == {
        TERMINATED_BY_TIMEOUT: 3,
        TERMINATED_BY_STOP_RECORDING: 1,
    }, stage["terminators"]
    # planned -> actual, because the terminator label is inferred: a stage
    # planned "manual" that ends any other way is a measurement worth seeing.
    assert stage["planned_vs_actual"]["manual->stop_recording"] == 1, stage


def test_heterogeneous_pool_is_named_not_refused() -> None:
    """Two arms may be pooled; pooling them unnoticed is the failure.

    A mock run reports robot_type "stage_runner_mock_robot" and holds a clean
    30 Hz by construction, so sweeping one into a glob beside real runs averages
    it into the same hertz_mean as the real ~20 Hz loop. Refusing to aggregate
    would be wrong -- comparing two arms is the point -- so the disagreement is
    named instead.
    """
    homogeneous = [make_trial("h1"), make_trial("h2")]
    assert heterogeneity(homogeneous) == {}, heterogeneity(homogeneous)
    mixed = [
        make_trial("h1"),
        make_trial("h2", robot_type="stage_runner_mock_robot", fps=60),
    ]
    differing = heterogeneity(mixed)
    assert set(differing) == {"robot_type", "fps"}, differing
    assert differing["fps"] == [30, 60], differing["fps"]
    summary = summarize(mixed, {})
    assert summary["heterogeneous"] == differing, summary["heterogeneous"]
    # Named, not refused: the timing is still computed over both.
    assert summary["timing"]["per_stage"]["s4"]["n"] == 2
    assert "pooled trials disagree" in format_report(summary)


def test_a_version_change_makes_the_pool_heterogeneous() -> None:
    """Same robot, same fps, same checkpoints -- different software.

    This is the axis the other four columns cannot see, and the one that already
    bit this package once: `gap_s` -> `stop_base_s` changed what a boundary
    number MEANS while robot_type, fps, declared_stage_ids and policy_paths all
    stayed identical. A pool spanning that change reports one mean over two
    definitions, and without these columns says nothing about it.

    Each axis is moved ALONE, because a test that changes all three at once
    passes even if two of the columns were never wired up.
    """
    moved_one_axis: tuple[tuple[str, TrialRecord, object], ...] = (
        (
            "stage_runner_version",
            make_trial("h2", stage_runner_version="0.2.0"),
            "0.2.0",
        ),
        ("lerobot_version", make_trial("h2", lerobot_version="0.5.0"), "0.5.0"),
        ("config_version", make_trial("h2", config_version=2), 2),
    )
    for column, odd_trial, odd_value in moved_one_axis:
        mixed = [make_trial("h1"), odd_trial]
        differing = heterogeneity(mixed)
        assert set(differing) == {column}, (column, differing)
        summary = summarize(mixed, {})
        assert "pooled trials disagree" in format_report(summary)
        # Traceable to a trial: the heterogeneity entry lists the distinct
        # values, not which trial held them.
        assert summary["trials"][1][column] == odd_value, summary["trials"][1]

    # UNKNOWN IS NOT A MATCH. An old log carrying no version fields pooled with a
    # new one is a real disagreement -- the None trial may well be from before
    # the change -- so it is named rather than assumed compatible.
    unknown = [make_trial("h1"), make_trial("h2", lerobot_version=None)]
    assert set(heterogeneity(unknown)) == {"lerobot_version"}, heterogeneity(unknown)
    # ... and an all-unknown pool is quiet: nothing was observed to differ.
    all_unknown = [
        make_trial("h1", lerobot_version=None),
        make_trial("h2", lerobot_version=None),
    ]
    assert heterogeneity(all_unknown) == {}, heterogeneity(all_unknown)


STANDALONE_HEADER = "stage_id,trials,successes,robot_type,fps,policy_path,measured_date"
STANDALONE_GOOD_ROW = "s5,10,8,mobileai_robot,30,mock://five,2026-09-10"


def test_standalone_rates_are_counts_and_are_validated() -> None:
    """The standalone file is hand-typed too, so it refuses the same errors.

    It is COUNTS rather than a rate because the report has to be able to say
    what the ratio's denominator rests on: 8/10 and 800/1000 are the same rate
    and not the same claim.

    And it is COUNTS PLUS PROVENANCE, because this row is joined to the chain on
    stage_id alone and then divided into it. A blank robot_type is not "the same
    robot", it is an unknown one, so a row missing any provenance column is
    rejected here rather than becoming a denominator whose conditions nobody can
    check. An unparseable measured_date is rejected for the same reason: the age
    comparison downstream would otherwise skip the axis it was meant to check.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "standalone.csv"
        path.write_text(
            f"{STANDALONE_HEADER},note\n{STANDALONE_GOOD_ROW},run alone\n",
            encoding="utf-8",
        )
        rates = load_standalone_rates(path)
        assert set(rates) == {"s5"}, rates
        assert rates["s5"].trials == 10 and rates["s5"].successes == 8
        assert close(rates["s5"].rate, 0.8), rates["s5"].rate
        assert rates["s5"].robot_type == "mobileai_robot", rates["s5"]
        assert rates["s5"].fps == 30, rates["s5"]
        assert rates["s5"].policy_path == "mock://five", rates["s5"]
        assert rates["s5"].measured_date == "2026-09-10", rates["s5"]

        path.write_text(
            f"{STANDALONE_HEADER}\n{STANDALONE_GOOD_ROW}\n"
            f"s5,4,2,mobileai_robot,30,mock://five,2026-09-10\n",
            encoding="utf-8",
        )
        try:
            load_standalone_rates(path)
        except ValueError as error:
            assert "duplicate" in str(error), error
        else:
            raise AssertionError("중복 standalone 행이 통과했다 / duplicate accepted")

        path.write_text(
            f"{STANDALONE_HEADER}\ns5,4,9,mobileai_robot,30,mock://five,2026-09-10\n",
            encoding="utf-8",
        )
        try:
            load_standalone_rates(path)
        except ValueError as error:
            assert "impossible" in str(error), error
        else:
            raise AssertionError("불가능한 카운트가 통과했다 / impossible counts")

        # The old three-column file: rejected by name, not silently defaulted to
        # "same conditions as whatever it is divided into".
        path.write_text("stage_id,trials,successes\ns5,10,8\n", encoding="utf-8")
        try:
            load_standalone_rates(path)
        except ValueError as error:
            message = str(error)
            assert "missing columns" in message, message
            assert "robot_type" in message and "measured_date" in message, message
        else:
            raise AssertionError(
                "provenance 없는 standalone 파일이 통과했다 / a file with no "
                "provenance columns was accepted"
            )

        # Columns present, one value blank: still an unknown condition.
        path.write_text(
            f"{STANDALONE_HEADER}\ns5,10,8,,30,mock://five,2026-09-10\n",
            encoding="utf-8",
        )
        try:
            load_standalone_rates(path)
        except ValueError as error:
            assert "robot_type" in str(error), error
        else:
            raise AssertionError("빈 provenance가 통과했다 / blank provenance accepted")

        path.write_text(
            f"{STANDALONE_HEADER}\ns5,10,8,mobileai_robot,30,mock://five,9월 10일\n",
            encoding="utf-8",
        )
        try:
            load_standalone_rates(path)
        except ValueError as error:
            assert "ISO date" in str(error), error
        else:
            raise AssertionError("잘못된 날짜가 통과했다 / non-ISO date accepted")


def test_standalone_from_another_condition_is_refused() -> None:
    """A denominator is only a denominator under the numerator's conditions.

    The join is on stage_id ALONE, so before this check a standalone rate
    measured on the mock robot, at 60 fps or on last month's checkpoint became
    the loss ratio's denominator without a word, and the quotient still printed
    as "the share of s5's standalone capability that survives the seam". It is
    not: it is the ratio between two different cells, and nothing about the
    number says so.

    Four independent trials (as in
    test_chaining_metrics_return_one_on_independent_stages):
        P(s5 | s4 ok) = 1/2 = 0.5   standalone 2/4 = 0.5   ratio = 1.0
    so the MATCHING row gives exactly 1.0 and every mismatch below refuses --
    the arithmetic is held constant so that only the provenance moves.
    """
    trials = [make_trial("x1"), make_trial("x2"), make_trial("x3"), make_trial("x4")]
    labels = {
        ("x1", "s4"): OUTCOME_SUCCESS,
        ("x1", "s5"): OUTCOME_SUCCESS,
        ("x2", "s4"): OUTCOME_SUCCESS,
        ("x2", "s5"): OUTCOME_FAIL,
        ("x3", "s4"): OUTCOME_FAIL,
        ("x3", "s5"): OUTCOME_SUCCESS,
        ("x4", "s4"): OUTCOME_FAIL,
        ("x4", "s5"): OUTCOME_FAIL,
    }
    stage_ids = ["s4", "s5"]

    matching = chaining_metrics(
        trials, labels, stage_ids, standalone_rates={"s5": standalone(4, 2)}
    )["chaining_loss_ratio"]
    assert matching["available"] is True, matching
    assert close(matching["value"], 1.0), matching["value"]
    assert matching["comparability"]["comparable"] is True, matching["comparability"]

    # Each axis, one at a time, with the counts unchanged.
    for label, rate in (
        ("robot_type", standalone(4, 2, robot_type="stage_runner_mock_robot")),
        ("fps", standalone(4, 2, fps=60)),
        ("policy_path", standalone(4, 2, policy_path="mock://four_checkpoints_ago")),
    ):
        refused = chaining_metrics(
            trials, labels, stage_ids, standalone_rates={"s5": rate}
        )["chaining_loss_ratio"]
        assert refused["available"] is False, (label, refused)
        assert refused["value"] is None, (label, refused)
        # The reason has to name the axis AND both values, because the operator
        # fixing this is looking at a CSV row, not at this module.
        assert label in refused["unavailable_reason"], (label, refused)
        assert len(refused["comparability"]["mismatches"]) == 1, refused
        assert label in refused["comparability"]["mismatches"][0], refused
        # And it says which measurement would produce a usable denominator.
        assert "SAME ROBOT" in refused["measurement_required"], refused

    # A pool that disagrees WITH ITSELF is a mismatch too: there is then no
    # single condition for the standalone row to match, and picking the trials
    # that agree with it would be choosing the denominator to suit the answer.
    # heterogeneity() names this pool; this refuses to divide into it.
    mixed = [*trials[:3], make_trial("x4", fps=60)]
    assert set(heterogeneity(mixed)) == {"fps"}, heterogeneity(mixed)
    into_mixed = chaining_metrics(
        mixed, labels, stage_ids, standalone_rates={"s5": standalone(4, 2)}
    )["chaining_loss_ratio"]
    assert into_mixed["available"] is False, into_mixed
    assert "fps" in into_mixed["unavailable_reason"], into_mixed

    # THE DATE IS THE ONE AXIS THAT CANNOT MATCH -- the standalone eval is a
    # separate session by construction -- so it warns and still divides.
    # 2026-08-01 to the trials' 2026-09-08 is 31 + 7 = 38 days, past the 14-day
    # review trigger.
    stale = chaining_metrics(
        trials,
        labels,
        stage_ids,
        standalone_rates={"s5": standalone(4, 2, measured_date="2026-08-01")},
    )["chaining_loss_ratio"]
    assert stale["available"] is True, stale
    assert close(stale["value"], 1.0), stale["value"]
    warnings = stale["comparability"]["warnings"]
    assert len(warnings) == 1 and "38 days" in warnings[0], warnings
    assert (
        stale["comparability"]["checked"]["measured_date"]["max_separation_days"] == 38
    ), stale["comparability"]["checked"]
    # Loudly: the warning is in the printed report, not only in the JSON.
    printed = format_report(
        summarize(trials, labels, {"s5": standalone(4, 2, measured_date="2026-08-01")})
    )
    assert "비교 조건 경고" in printed and "38 days" in printed, printed

    # AN AXIS THE CHAIN CANNOT ANSWER IS NOT A MISMATCH. A log written before the
    # provenance fields existed carries no robot_type, so there is nothing to
    # disagree with -- it lands in `unchecked` and the ratio is still computed,
    # the same way a missing label shrinks a denominator instead of counting as a
    # failure. "Could not check" and "checked and the same" must not print alike.
    unprovenanced = [
        TrialRecord(
            run_id=trial.run_id,
            started_iso=trial.started_iso,
            ended_iso=trial.ended_iso,
            stages=trial.stages,
            transition_gaps_s=trial.transition_gaps_s,
            stop_base_durations_s=trial.stop_base_durations_s,
            stop_base_fallback_stage_ids=(),
            stop_base_failed_stage_ids=(),
            completed=True,
            end_reason=TRIAL_REASON_COMPLETED,
        )
        for trial in trials
    ]
    old_log = chaining_metrics(
        unprovenanced, labels, stage_ids, standalone_rates={"s5": standalone(4, 2)}
    )["chaining_loss_ratio"]
    assert old_log["available"] is True, old_log
    assert old_log["comparability"]["unchecked"] == [
        "robot_type",
        "fps",
        "policy_path",
    ], old_log["comparability"]
    assert old_log["comparability"]["mismatches"] == [], old_log["comparability"]
    old_printed = format_report(
        summarize(unprovenanced, labels, {"s5": standalone(4, 2)})
    )
    assert "not checked" in old_printed, old_printed


def test_truncated_trial_is_incomplete() -> None:
    trials, labels = load_fixture()
    truncated = trials[2]
    assert truncated.run_id == "t03", truncated.run_id
    # No trial_end line at all: the process was killed. The stage that DID
    # finish survives, because its measurements are real.
    assert truncated.completed is False, truncated
    assert truncated.end_reason == TRIAL_REASON_TRUNCATED, truncated.end_reason
    assert truncated.ended_iso is None, truncated.ended_iso
    assert len(truncated.stages) == 1, truncated.stages
    assert truncated.stages[0].terminated_by == TERMINATED_BY_STOP_RECORDING
    assert truncated.transition_gaps_s == (), truncated.transition_gaps_s
    # Its one boundary IS logged (to_stage_id null, after the aborted stage) but
    # the stop_base call raised there, so it contributes no duration and one
    # named failure instead. Before the schema change this trial produced no
    # transition line at all, and a base left driving looked like a clean run.
    assert truncated.stop_base_durations_s == (), truncated.stop_base_durations_s
    assert truncated.stop_base_fallback_stage_ids == (), truncated
    assert truncated.stop_base_failed_stage_ids == ("s4",), truncated
    # It is excluded from every success metric, and summarize() says so by name
    # rather than just shrinking a denominator.
    summary = summarize(trials, labels)
    assert summary["scored_trial_count"] == 2, summary["scored_trial_count"]
    assert summary["excluded_run_ids"] == ["t03"], summary["excluded_run_ids"]
    # A stage_start whose stage_end never arrived (killed mid-stage) yields no
    # StageRecord: there is no elapsed_s, no frame count and no terminator to
    # invent. The trial is still readable.
    dangling = build_trial(
        [
            {
                "event": "trial_start",
                "schema_version": 1,
                "run_id": "t04",
                "wall_clock_iso": "2026-09-08T16:00:00.000+09:00",
                "stage_id": None,
                "frame_idx": 0,
                "terminator": None,
                "reason": "",
            },
            {
                "event": "stage_start",
                "schema_version": 1,
                "run_id": "t04",
                "wall_clock_iso": "2026-09-08T16:00:01.000+09:00",
                "stage_id": "s4",
                "frame_idx": 0,
                "terminator": "timeout",
                "reason": "",
                "stage_index": 0,
            },
        ]
    )
    assert dangling.stages == (), dangling.stages
    assert dangling.completed is False and dangling.end_reason == TRIAL_REASON_TRUNCATED


def test_metrics_without_labels_are_unavailable() -> None:
    trials, _ = load_fixture()
    summary = summarize(trials, {})
    # Timing needs no human input and is still there.
    assert summary["timing"]["per_stage"]["s4"]["n"] == 3
    # Success does. Every success metric is None, never 0.0 and never guessed
    # from terminators -- a stage that timed out cleanly is not a success, and
    # "not measured" must not print the same as "measured as zero".
    assert summary["labels_available"] is False
    for key in (
        "per_stage_success_rate",
        "chain_success_rate",
        "average_sequence_length",
        "failure_histogram",
        "missing_labels",
        "chaining",
    ):
        assert summary[key] is None, f"{key} = {summary[key]!r}"
    assert "labels file" in summary["note"], summary["note"]
    # And with labels present the same call fills them in.
    _, labels = load_fixture()
    assert summarize(trials, labels)["chain_success_rate"] == [1.0, 0.5]


def test_aborted_boundary_contributes_no_gap() -> None:
    """A boundary that leads nowhere must not enter the gap sample.

    The runner writes one of these after every abort and after the last stage
    (to_stage_id null), because a stop_base that silently failed must not produce
    a log identical to a clean run. Deriving the gap from stage_end -> next
    stage_start is what makes the reader indifferent to the shape: no stage_start
    follows, so there is no gap to measure, while the stop_base call that DID
    happen is still counted.

    Both shapes are asserted, since old logs carry the other one: a boundary
    whose to_stage_id names a stage that never started (the frozen step order
    16 f emit, then 16 g break) must behave identically.
    """
    base = {
        "schema_version": 1,
        "run_id": "t05",
        "stage_id": None,
        "frame_idx": 0,
        "terminator": None,
        "reason": "",
    }
    trial = build_trial(
        [
            {
                **base,
                "event": "trial_start",
                "wall_clock_iso": "2026-09-08T18:00:00.000+09:00",
            },
            {
                **base,
                "event": "stage_start",
                "wall_clock_iso": "2026-09-08T18:00:01.000+09:00",
                "stage_id": "s4",
                "terminator": TERMINATED_BY_TIMEOUT,
                "stage_index": 0,
            },
            {
                **base,
                "event": "stage_end",
                "wall_clock_iso": "2026-09-08T18:00:04.200+09:00",
                "stage_id": "s4",
                "frame_idx": 94,
                "terminator": TERMINATED_BY_STOP_RECORDING,
                "stage_index": 0,
                "elapsed_s": 3.2,
                "frames": 94,
                "hertz": 29.375,
            },
            {
                **base,
                "event": "transition",
                "wall_clock_iso": "2026-09-08T18:00:04.229+09:00",
                "stage_id": "s4",
                "frame_idx": 94,
                "reason": "stop_base",
                "from_stage_id": "s4",
                # null: nothing followed. An older log names "s5" here, which
                # this reader must treat the same way -- neither is a crossing.
                "to_stage_id": None,
                "stop_action": {"x.vel": 0.0, "theta.vel": 0.0},
                "stop_base_s": 0.029,
            },
            {
                **base,
                "event": "trial_end",
                "wall_clock_iso": "2026-09-08T18:00:07.900+09:00",
                "frame_idx": 94,
                "reason": "aborted",
                "completed": False,
                "stage_count": 1,
                "elapsed_s": 7.9,
            },
        ]
    )
    assert trial.transition_gaps_s == (), trial.transition_gaps_s
    assert trial.stop_base_durations_s == (0.029,), trial.stop_base_durations_s
    assert trial.stop_base_fallback_stage_ids == (), trial
    assert trial.stop_base_failed_stage_ids == (), trial.stop_base_failed_stage_ids
    timing = timing_summary([trial])
    residual = timing["stage_end_to_stage_start_residual_s"]
    assert residual["n"] == 0, residual
    assert timing["stop_base_s"]["n"] == 1, timing["stop_base_s"]
    assert timing["stop_base_failed_n"] == 0, timing["stop_base_failed_n"]


def test_a_non_primary_stop_base_has_no_boundary_total() -> None:
    """A crossed boundary can still have no total, and the count says so.

    The two halves have different sample sizes on purpose: a boundary that leads
    nowhere has no residual, and a boundary whose stop_base fell back or failed
    has no usable first half -- its stop_base_s is how long the failing
    get_observation took, not what stopping the base cost. Adding the two
    reported MEANS would silently paper over both, which is why build_trial pairs
    them per boundary instead.

    One trial, two stages, and the crossing is the broken one:

        s4 stage_start 19:00:01.000 + 3.000 + 0.500 (FALLBACK) + 0.010
                                                     -> stage_end 19:00:04.510
           transition  19:00:04.512 -> s5
        s5 stage_start 19:00:04.515  -> residual 0.005 s, and NO total
                       + 2.000 + 0.020 (primary) + 0.010 -> stage_end 06.545
           transition  19:00:06.547 -> null (no crossing, so no residual either)

    So: one residual, one primary stop_base duration, and zero boundary costs --
    the residual is still real (the emits did happen) but nothing may be added to
    it, and `unpaired_boundary_n` is what stops that reading as "the boundary
    cost 0.005 s".
    """
    base = {
        "schema_version": 1,
        "run_id": "t07",
        "stage_id": None,
        "frame_idx": 0,
        "terminator": None,
        "reason": "",
    }
    trial = build_trial(
        [
            {
                **base,
                "event": "trial_start",
                "wall_clock_iso": "2026-09-08T19:00:00.000+09:00",
            },
            {
                **base,
                "event": "stage_start",
                "wall_clock_iso": "2026-09-08T19:00:01.000+09:00",
                "stage_id": "s4",
                "terminator": TERMINATED_BY_TIMEOUT,
                "stage_index": 0,
            },
            {
                **base,
                "event": "stage_end",
                "wall_clock_iso": "2026-09-08T19:00:04.510+09:00",
                "stage_id": "s4",
                "frame_idx": 90,
                "terminator": TERMINATED_BY_TIMEOUT,
                "stage_index": 0,
                "elapsed_s": 3.0,
                "frames": 90,
                "hertz": 30.0,
            },
            {
                **base,
                "event": "transition",
                "wall_clock_iso": "2026-09-08T19:00:04.512+09:00",
                "stage_id": "s4",
                "frame_idx": 90,
                "reason": "stop_base_fallback",
                "from_stage_id": "s4",
                "to_stage_id": "s5",
                "stop_base_path": "fallback",
                "stop_action": None,
                "stop_base_s": 0.5,
            },
            {
                **base,
                "event": "stage_start",
                "wall_clock_iso": "2026-09-08T19:00:04.515+09:00",
                "stage_id": "s5",
                "frame_idx": 90,
                "terminator": TERMINATED_BY_MANUAL,
                "stage_index": 1,
            },
            {
                **base,
                "event": "stage_end",
                "wall_clock_iso": "2026-09-08T19:00:06.545+09:00",
                "stage_id": "s5",
                "frame_idx": 150,
                "terminator": TERMINATED_BY_MANUAL,
                "stage_index": 1,
                "elapsed_s": 2.0,
                "frames": 60,
                "hertz": 30.0,
            },
            {
                **base,
                "event": "transition",
                "wall_clock_iso": "2026-09-08T19:00:06.547+09:00",
                "stage_id": "s5",
                "frame_idx": 150,
                "reason": "stop_base",
                "from_stage_id": "s5",
                "to_stage_id": None,
                "stop_base_path": "primary",
                "stop_action": {"x.vel": 0.0, "theta.vel": 0.0},
                "stop_base_s": 0.02,
            },
            {
                **base,
                "event": "trial_end",
                "wall_clock_iso": "2026-09-08T19:00:09.000+09:00",
                "frame_idx": 150,
                "reason": TRIAL_REASON_COMPLETED,
                "completed": True,
                "stage_count": 2,
                "elapsed_s": 9.0,
            },
        ]
    )
    # 04.515 - 04.510 = 0.005 s, and it is NOT joined to the 0.5 s beside it.
    assert trial.transition_gaps_s == (0.005,), trial.transition_gaps_s
    assert trial.stop_base_durations_s == (0.02,), trial.stop_base_durations_s
    assert trial.stop_base_fallback_stage_ids == ("s4",), trial
    assert trial.boundary_costs_s == (), trial.boundary_costs_s

    timing = timing_summary([trial])
    boundary_cost = timing["boundary_cost_s"]
    assert boundary_cost["n"] == 0, boundary_cost
    assert boundary_cost["crossed_boundary_n"] == 1, boundary_cost
    assert boundary_cost["unpaired_boundary_n"] == 1, boundary_cost
    residual = timing["stage_end_to_stage_start_residual_s"]
    assert residual["n"] == 1, residual
    # The shortfall is printed, not left to be inferred from two n values.
    report = format_report(summarize([trial], {}))
    assert "1 crossed boundaries have no total" in report, report


def test_stop_action_null_is_not_evidence_of_a_failed_base_stop() -> None:
    """The fallback and the failure look identical in stop_action, and are not.

    transitions.stop_base returns no action dict on EITHER of its non-primary
    paths: the fallback deliberately does not build one (it calls
    base.set_cmd_vel(0, 0) directly, which is the whole point of it), and the
    failure never got one built. So `stop_action: null` covers a base that IS
    stopped and a base that may still be driving, and a reader that keys off it
    reports a working safety mechanism as a hazard -- or hides a hazard inside a
    bucket the operator has learned to ignore. stop_base_path is the field that
    separates them, and `reason` is the pre-field fallback.
    """
    base = {
        "schema_version": 1,
        "run_id": "t06",
        "stage_id": "s4",
        "frame_idx": 0,
        "terminator": None,
        "stop_action": None,
        "from_stage_id": "s4",
        "to_stage_id": None,
        "stop_base_s": 0.734,
        "wall_clock_iso": "2026-09-08T20:00:04.000+09:00",
        "event": "transition",
    }
    fallback = stop_base_path({**base, "reason": "stop_base_fallback"})
    failed = stop_base_path({**base, "reason": "stop_base_failed"})
    assert fallback == "fallback" and failed == "failed", (fallback, failed)
    # The explicit field wins over the reason string, since the writer derives
    # the string from it.
    assert (
        stop_base_path(
            {**base, "reason": "stop_base_failed", "stop_base_path": "fallback"}
        )
        == "fallback"
    )
    # Wording this reader does not know is filed as a FAILURE, not as a clean
    # stop: over-reporting costs a line in the report, under-reporting costs a
    # base that drove through save_episode with nothing saying so.
    assert stop_base_path({**base, "reason": "stop_base_gave_up"}) == "failed"


def test_unknown_outcome_is_rejected_not_counted_as_a_failure() -> None:
    """A typo in the outcome column must stop the aggregation, not shrink it.

    Everything downstream tests `== OUTCOME_SUCCESS`, so before this check a
    row typed as "succes" was filed as a FAILURE: chain_success_rate and
    average_sequence_length dropped and the first-failure histogram gained a
    failure the operator never wrote down, with nothing printed anywhere. This
    file is typed by hand off a paper sheet, so that row is the expected error.
    """
    assert OUTCOME_VALUES == (OUTCOME_SUCCESS, OUTCOME_FAIL), OUTCOME_VALUES
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "labels.csv"
        path.write_text(
            "run_id,stage_id,outcome,failure_phase\nt01,s4,success,\nt01,s5,succes,\n",
            encoding="utf-8",
        )
        try:
            load_labels(path)
        except ValueError as error:
            message = str(error)
        else:
            raise AssertionError("오타 outcome이 통과했다 / typo accepted as a label")
        # The message has to name the value, the file and the row, and the
        # accepted vocabulary, because the person reading it is looking at a
        # CSV, not at this module. Row 3, counting the header as row 1.
        assert "succes" in message, message
        assert f"{path}:3" in message, message
    assert OUTCOME_SUCCESS in message and OUTCOME_FAIL in message, message
    # Case and surrounding blanks are still normalised, not rejected.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "labels.csv"
        path.write_text("run_id,stage_id,outcome\nt01,s4, SUCCESS \n", encoding="utf-8")
        assert load_labels(path) == {("t01", "s4"): OUTCOME_SUCCESS}


def test_aggregate_imports_without_the_robot_stack() -> None:
    """The stdlib-only claim in aggregate.py's docstring, actually checked.

    The module itself only ever imported stdlib plus `results`, but importing it
    executes the package __init__ first, and while that was an eager re-export
    list the import chain reached draccus, lerobot and torch -- so `python -m
    stage_runner.aggregate old_run/events.jsonl` failed on any machine without
    the robot stack, which is the one machine the aggregator was put in the
    package for. Run in a SUBPROCESS on purpose: under pytest another test in
    the same process may already have imported lerobot, and then this assertion
    would pass or fail for reasons that have nothing to do with the property.
    """
    probe = (
        "import sys, stage_runner.aggregate;"
        "loaded = [name for name in ('torch', 'lerobot', 'draccus', 'rerun')"
        " if name in sys.modules];"
        "print(loaded);"
        "sys.exit(1 if loaded else 0)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(TESTS_DIRECTORY.parent / "src")
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "aggregate가 로봇 스택을 끌고 온다 / aggregate dragged in: "
        f"{completed.stdout.strip()}{completed.stderr.strip()}"
    )


def make_trial(
    run_id: str,
    robot_type: str = "mobileai_robot",
    fps: int = 30,
    stage_runner_version: str | None = "0.1.0",
    lerobot_version: str | None = "0.4.4",
    config_version: int | None = 1,
) -> TrialRecord:
    """A completed two-stage trial carrying no timing worth asserting.

    The success metrics read only run_id, completed and the labels, so the
    timing fields are filler here; the fixture is where timing is checked.
    robot_type, fps and the three version fields are parameters only so the
    heterogeneity check has two trials that legitimately disagree.
    """
    stages = tuple(
        StageRecord(
            stage_id=stage_id,
            stage_index=stage_index,
            planned_terminator=TERMINATED_BY_TIMEOUT,
            terminated_by=TERMINATED_BY_TIMEOUT,
            reason="",
            elapsed_s=1.0,
            frames=30,
            start_frame_idx=30 * stage_index,
            end_frame_idx=30 * (stage_index + 1),
            start_wall_clock_iso="2026-09-08T17:00:00.000+09:00",
            end_wall_clock_iso="2026-09-08T17:00:01.000+09:00",
        )
        for stage_index, stage_id in enumerate(("s4", "s5"))
    )
    return TrialRecord(
        run_id=run_id,
        started_iso="2026-09-08T17:00:00.000+09:00",
        ended_iso="2026-09-08T17:00:02.000+09:00",
        stages=stages,
        # Filler, but filler in the shape the writer now produces: the residual
        # is the small leftover AFTER stop_base, and the total is their sum
        # (0.031 + 0.003 = 0.034). Copying the old shape from here is how the
        # inverted relation would get reintroduced.
        transition_gaps_s=(0.003,),
        stop_base_durations_s=(0.031, 0.029),
        boundary_costs_s=(0.034,),
        stop_base_fallback_stage_ids=(),
        stop_base_failed_stage_ids=(),
        completed=True,
        end_reason=TRIAL_REASON_COMPLETED,
        robot_type=robot_type,
        fps=fps,
        dataset_repo_id=f"local/eval_{run_id}",
        declared_stage_ids=("s4", "s5"),
        policies=(("s4", "mock://four"), ("s5", "mock://five")),
        stage_runner_version=stage_runner_version,
        lerobot_version=lerobot_version,
        config_version=config_version,
    )


def make_single_stage_trial(
    run_id: str,
    *,
    elapsed_s: float,
    frames: int,
    terminated_by: str = TERMINATED_BY_TIMEOUT,
    planned_terminator: str = TERMINATED_BY_TIMEOUT,
) -> TrialRecord:
    """One trial whose only stage is s5, for the timing exclusion rules.

    elapsed_s=0.0, frames=0 is exactly what executors._aborted_before_start
    returns when the operator aborts inside the inter-stage gap: the runner
    emits a full stage_end for a stage that never entered record_loop.
    """
    stage = StageRecord(
        stage_id="s5",
        stage_index=0,
        planned_terminator=planned_terminator,
        terminated_by=terminated_by,
        reason="",
        elapsed_s=elapsed_s,
        frames=frames,
        start_frame_idx=0,
        end_frame_idx=frames,
        start_wall_clock_iso="2026-09-08T19:00:00.000+09:00",
        end_wall_clock_iso="2026-09-08T19:00:09.000+09:00",
    )
    return TrialRecord(
        run_id=run_id,
        started_iso="2026-09-08T19:00:00.000+09:00",
        ended_iso="2026-09-08T19:00:10.000+09:00",
        stages=(stage,),
        transition_gaps_s=(),
        stop_base_durations_s=(),
        stop_base_fallback_stage_ids=(),
        stop_base_failed_stage_ids=(),
        completed=True,
        end_reason=TRIAL_REASON_COMPLETED,
    )


if __name__ == "__main__":
    tests = [
        test_build_trial_reads_two_stages,
        test_hertz_is_frames_over_elapsed,
        test_timing_summary_includes_the_truncated_trial,
        test_per_stage_and_chain_success_rate,
        test_average_sequence_length,
        test_failure_histogram_counts_first_failure,
        test_chaining_metrics_on_the_fixture,
        test_chaining_metrics_return_one_on_independent_stages,
        test_in_chain_rate_is_never_used_as_the_standalone_denominator,
        test_missing_label_shrinks_one_denominator_and_is_counted,
        test_stage_that_never_ran_is_not_a_slow_stage,
        test_heterogeneous_pool_is_named_not_refused,
        test_a_version_change_makes_the_pool_heterogeneous,
        test_standalone_rates_are_counts_and_are_validated,
        test_standalone_from_another_condition_is_refused,
        test_truncated_trial_is_incomplete,
        test_metrics_without_labels_are_unavailable,
        test_aborted_boundary_contributes_no_gap,
        test_a_non_primary_stop_base_has_no_boundary_total,
        test_stop_action_null_is_not_evidence_of_a_failed_base_stop,
        test_unknown_outcome_is_rejected_not_counted_as_a_failure,
        test_aggregate_imports_without_the_robot_stack,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print(f"[done] {len(tests)} passed")
