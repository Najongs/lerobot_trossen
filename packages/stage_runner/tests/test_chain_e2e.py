"""The whole chain, end to end, with no hardware and no robot SDK.

``cli.main`` -> preflight -> dataset -> bundles -> ``run_trial`` -> 22 stages ->
``save_episode``, driven by ``MockRobot`` and ``mock://progress``. Same code
path the robot PC takes, with ``STAGE_RUNNER_NO_PLUGINS=1`` so the third-party
plugin scan (and through it ``trossen_slate``) never runs.

Three runs:

* ``mock://progress`` -- all 11 stages complete automatically, with 11 resets
  (10 of them at a boundary). The success case.
* ``mock://stuck``    -- stage 1's progress never rises, the stage runs to
  p90 x timeout_factor, the chain FAILS and stops. No retry, no later stage.
* an injected right arrow -- the stage ends with ``terminated_by="manual"``,
  which the chain accepts and counts separately from ``"complete"``.

``record_loop`` sleeps in REAL time, so every duration in
``configs/chain_mock.yaml`` and ``data/stage_params_mock.json`` is shrunk. The
arithmetic that must hold at the REAL 21 Hz / 1.5 s / 0.7 rad is asserted in
``test_chain_unit.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

os.environ.setdefault("STAGE_RUNNER_NO_PLUGINS", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_SRC = REPO_ROOT / "packages" / "stage_runner" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

TESTS = Path(__file__).resolve().parent
CHAIN_CONFIG = TESTS / "configs" / "chain_mock.yaml"
MOCK_PARAMS = TESTS / "data" / "stage_params_mock.json"

from stage_runner import cli, config, record_adapter, transitions  # noqa: E402
from stage_runner.results import (  # noqa: E402
    TERMINATED_BY_COMPLETE,
    TERMINATED_BY_MANUAL,
    TERMINATED_BY_REACHED,
    TERMINATED_BY_TIMEOUT,
    TRIAL_REASON_CHAIN_FAILED,
    TRIAL_REASON_COMPLETED,
)

FORBIDDEN_MODULES = ("lerobot_robot_trossen", "trossen_slate")

EXPECTED_STAGE_IDS = [
    "reset_pre_01",
    "t01",
    "reset_to_02",
    "t02",
    "reset_to_03",
    "t03",
    "reset_to_04",
    "t04",
    "reset_to_05",
    "t05",
    "reset_to_06",
    "t06",
    "reset_to_07",
    "t07",
    "reset_to_08",
    "t08",
    "reset_to_09",
    "t09",
    "reset_to_10",
    "t10",
    "reset_to_11",
    "t11",
]


class ChainRun:
    """One ``cli.main`` invocation and everything it left on disk."""

    def __init__(self, directory: Path, exit_code: int, output_root: Path) -> None:
        self.directory = directory
        self.exit_code = exit_code
        self.output_root = output_root
        self.events = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]

    def of(self, event: str) -> list[dict]:
        return [record for record in self.events if record["event"] == event]

    @property
    def trial_end(self) -> dict:
        return self.of("trial_end")[-1]


def run_chain(
    temporary: Path,
    *,
    policy_path: str = "mock://progress",
    to_stage: int = 11,
    extra_argv: tuple[str, ...] = (),
    patch_events=None,
) -> ChainRun:
    dataset_root = temporary / "dataset"
    output_root = temporary / "outputs"
    argv = [
        "--config_path",
        str(CHAIN_CONFIG),
        f"--chain.params_path={MOCK_PARAMS}",
        f"--chain.model.policy_path={policy_path}",
        f"--chain.to_stage={to_stage}",
        f"--dataset.root={dataset_root}",
        f"--output.root={output_root}",
        *extra_argv,
    ]

    original = record_adapter.make_keyboard_events
    if patch_events is not None:
        record_adapter.make_keyboard_events = patch_events
    try:
        exit_code = cli.main(argv)
    finally:
        record_adapter.make_keyboard_events = original

    assert output_root.exists(), (
        f"cli.main returned {exit_code} without creating a run directory "
        "(it exited before step 9, i.e. preflight refused the run -- read the "
        "ERROR line above this traceback)"
    )
    directories = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert len(directories) == 1, directories
    return ChainRun(directories[0], exit_code, output_root)


class ChainEndToEndTest(unittest.TestCase):
    def tearDown(self) -> None:  # noqa: N802
        leaked = [name for name in FORBIDDEN_MODULES if name in sys.modules]
        self.assertEqual(
            leaked,
            [],
            "the chain must run end to end with no robot SDK: "
            "STAGE_RUNNER_NO_PLUGINS=1 must keep register_plugins() from "
            "importing lerobot_robot_trossen (and through it trossen_slate)",
        )

    # ------------------------------------------------------------------ success

    def test_eleven_stages_complete_with_eleven_resets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(Path(directory))

        self.assertEqual(run.exit_code, cli.EXIT_OK)

        starts = run.of("stage_start")
        ends = run.of("stage_end")
        self.assertEqual([event["stage_id"] for event in starts], EXPECTED_STAGE_IDS)
        self.assertEqual([event["stage_id"] for event in ends], EXPECTED_STAGE_IDS)

        policy_ends = [event for event in ends if event["kind"] == "policy"]
        reset_ends = [event for event in ends if event["kind"] == "reset"]
        self.assertEqual(len(policy_ends), 11)
        self.assertEqual(
            len(reset_ends),
            11,
            "an 11-stage chain has ELEVEN ramps: ten boundaries plus the "
            "initial reset. The brief's '10 resets' counts the boundaries",
        )

        for event in policy_ends:
            self.assertEqual(
                event["terminator"],
                TERMINATED_BY_COMPLETE,
                f"stage {event['stage_id']} did not complete automatically: "
                f"{event['reason']}",
            )
            detail = event["reason_detail"]
            self.assertEqual(detail["completion_reason"], "progress")
            self.assertEqual(detail["p_last"], 1.0)
            self.assertGreaterEqual(detail["elapsed_s"], detail["p10_s"])
            # The departure latch is a premise of every completion, and the
            # report's `출발` column reads these two keys.
            self.assertTrue(
                detail["departed"],
                f"stage {event['stage_id']} completed without ever leaving its "
                "start pose, which the monitor must refuse",
            )
            self.assertIsNotNone(detail["departed_s"])
            self.assertLessEqual(detail["departed_s"], detail["elapsed_s"])
        for event in reset_ends:
            self.assertEqual(
                event["terminator"],
                TERMINATED_BY_REACHED,
                f"reset {event['stage_id']} did not arrive: {event['reason']}",
            )
            detail = event["reason_detail"]
            self.assertEqual(detail["reset_end_reason"], "reached")
            self.assertLess(detail["reach_err"], detail["reset_tol_rad"])
            self.assertLessEqual(
                detail["reset_max_step_rad"],
                0.1,
                "the ramp's per-tick step must stay inside the 0.1 rad "
                "relative-target clamp; the clamp is a safety net, not the "
                "mechanism",
            )

        # Planned terminators, as the aggregator reads them off stage_start.
        self.assertEqual(
            {event["terminator"] for event in starts if event["kind"] == "policy"},
            {"completion"},
        )
        self.assertEqual(
            {event["terminator"] for event in starts if event["kind"] == "reset"},
            {"reached"},
        )

        # stage_number is the chain's number; stage_index is not.
        last = ends[-1]
        self.assertEqual((last["stage_number"], last["stage_index"]), (11, 21))

        trial_end = run.trial_end
        self.assertEqual(trial_end["reason"], TRIAL_REASON_COMPLETED)
        self.assertTrue(trial_end["completed"])
        self.assertTrue(trial_end["episode_saved"])
        self.assertFalse(trial_end["chain_failed"])
        self.assertEqual(trial_end["stage_count"], 22)
        self.assertEqual(trial_end["stages_configured"], 22)
        self.assertGreater(trial_end["buffered_frames"], 0)

    def test_one_episode_holds_every_stage_and_the_tasks_change(self) -> None:
        """One chain = ONE episode. A per-stage save would destroy the transition."""
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(Path(directory), to_stage=3)
            dataset_root = Path(directory) / "dataset"
            info = json.loads((dataset_root / "meta" / "info.json").read_text())
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(
                info["features"]["action"]["shape"],
                [17],
                "has_progress: true must widen the recorded action to 17, which "
                "is the same declaration that makes make_policy build a 17-wide "
                "head for a `tph` checkpoint",
            )
            self.assertEqual(info["features"]["action"]["names"][-1], "progress")
            tasks = json.loads(
                (dataset_root / "meta" / "tasks.jsonl").read_text()
                .splitlines()[0]
            ) if (dataset_root / "meta" / "tasks.jsonl").exists() else None
        self.assertEqual(run.exit_code, cli.EXIT_OK)
        # The reset frames carry their own task string, which is how a boundary
        # is found in the dataset afterwards.
        reset_tasks = {
            event["single_task"]
            for event in run.of("stage_start")
            if event["kind"] == "reset"
        }
        self.assertEqual(reset_tasks, {"reset:01", "reset:02", "reset:03"})
        _ = tasks  # presence only; the schema of tasks.jsonl is lerobot's

    def test_a_boundary_stops_the_base_after_every_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(Path(directory), to_stage=2)
        boundaries = run.of("transition")
        # One per stage INCLUDING the last: save_episode flushes encoders and
        # the base would hold its last command for all of it.
        self.assertEqual(len(boundaries), len(run.of("stage_end")))
        for boundary in boundaries:
            self.assertEqual(boundary["stop_base_path"], transitions.STOP_PATH_PRIMARY)
            self.assertEqual(boundary["stop_action"]["x.vel"], 0.0)
            self.assertEqual(boundary["stop_action"]["theta.vel"], 0.0)
        self.assertIsNone(
            boundaries[-1]["to_stage_id"], "the last boundary leads nowhere"
        )

    # ------------------------------------------------------------------ failure

    def test_a_stuck_stage_times_out_and_breaks_the_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(Path(directory), policy_path="mock://stuck")

        self.assertEqual(run.exit_code, cli.EXIT_CHAIN_FAILED)

        ends = run.of("stage_end")
        # The initial reset, then stage 1 -- and NOTHING after it.
        self.assertEqual(
            [event["stage_id"] for event in ends],
            ["reset_pre_01", "t01"],
            "a chain failure stops the chain: there is no retry and no later "
            "stage runs",
        )
        self.assertEqual(ends[0]["terminator"], TERMINATED_BY_REACHED)
        self.assertEqual(ends[1]["terminator"], TERMINATED_BY_TIMEOUT)
        self.assertEqual(
            ends[1]["required_terminator"],
            [TERMINATED_BY_COMPLETE, TERMINATED_BY_MANUAL],
        )
        # `stuck` holds the arm AND the base, so it also never departs -- and a
        # timeout with no departure is a different finding from an ordinary one:
        # the stage did not run out of time doing its task, it never started.
        self.assertFalse(ends[1]["reason_detail"]["departed"])
        self.assertIsNone(ends[1]["reason_detail"]["departed_s"])
        self.assertIn(
            "never_departed",
            ends[1]["reason"],
            "the reason must name it, because that is what tells the operator "
            "to look at the start scene instead of at the policy",
        )
        self.assertEqual(
            ends[0]["reason_detail"]["reset_initial"],
            True,
            "the first ramp is the INITIAL one, held to initial_max_jump_rad",
        )
        self.assertEqual(
            ends[0]["reason_detail"]["reset_jump_limit_rad"],
            0.6,
            "chain_mock.yaml leaves initial_max_jump_rad at its default, and "
            "the initial ramp must be held to THAT and not to max_jump_rad",
        )

        # The base was still stopped at the failing boundary.
        boundaries = run.of("transition")
        self.assertEqual(len(boundaries), 2)
        self.assertEqual(boundaries[-1]["stop_base_path"], transitions.STOP_PATH_PRIMARY)
        self.assertIsNone(boundaries[-1]["to_stage_id"])

        trial_end = run.trial_end
        self.assertEqual(trial_end["reason"], TRIAL_REASON_CHAIN_FAILED)
        self.assertFalse(trial_end["completed"])
        self.assertTrue(trial_end["chain_failed"])
        self.assertEqual(trial_end["chain_failed_stage_id"], "t01")
        self.assertTrue(
            trial_end["episode_saved"],
            "a partial chain IS data: the episode is saved with "
            "completed=false, and discarding it would be irreversible",
        )
        self.assertEqual(trial_end["stages_configured"], 22)
        self.assertEqual(trial_end["stage_count"], 2)

    def test_a_manual_completion_is_accepted_and_counted_separately(self) -> None:
        """The right arrow ends the stage; the chain goes on, and says which it was.

        The press is INJECTED rather than typed: the keyboard listener is
        core's, this machine has no display, and what is under test is the
        classification -- not pynput.
        """
        state = {"armed": False}

        def keyboard_events():
            events = {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
            }
            state["events"] = events
            return None, events

        with tempfile.TemporaryDirectory() as directory:
            # `stuck` never completes on its own, so an accepted `manual` can
            # only have come from the injected press.
            import threading

            def press_right_arrow() -> None:
                # Wait for the first policy stage to be running: the initial
                # reset has to finish first, or the press would be discarded in
                # the inter-stage gap (which is itself the behaviour
                # _discard_stale_exit_early exists for).
                import time

                deadline = time.perf_counter() + 20.0
                while time.perf_counter() < deadline:
                    events = state.get("events")
                    if events is not None and state["armed"]:
                        events["exit_early"] = True
                        return
                    time.sleep(0.01)

            thread = threading.Thread(target=press_right_arrow, daemon=True)

            original_call = record_adapter.call_record_loop

            def call_record_loop(**kwargs):
                # Arm the press once a POLICY stage's loop has been entered.
                state["armed"] = "reset" not in str(kwargs.get("single_task", ""))
                return original_call(**kwargs)

            record_adapter.call_record_loop = call_record_loop
            thread.start()
            try:
                run = run_chain(
                    Path(directory),
                    policy_path="mock://stuck",
                    to_stage=1,
                    patch_events=keyboard_events,
                )
            finally:
                record_adapter.call_record_loop = original_call

        ends = run.of("stage_end")
        stage_one = next(event for event in ends if event["stage_id"] == "t01")
        self.assertEqual(
            stage_one["terminator"],
            TERMINATED_BY_MANUAL,
            "an early end with no monitor result is a right-arrow press, and it "
            f"must be recorded as `manual`, never as `complete`: "
            f"{stage_one['reason']}",
        )
        self.assertNotEqual(stage_one["terminator"], TERMINATED_BY_COMPLETE)
        self.assertIn("elapsed_s", stage_one["reason"])
        # A manual completion is in required_terminator, so the chain did NOT
        # fail -- that is the user's 2026-10-06 decision, and the distinction
        # survives in the terminator value.
        self.assertFalse(run.trial_end["chain_failed"])
        self.assertEqual(run.exit_code, cli.EXIT_OK)

    def test_reset_only_runs_eleven_ramps_and_no_policy(self) -> None:
        """bring-up (2)/(3): the ramps alone, with no checkpoint loaded at all.

        A procedure ("press ESC right after `reached`") cannot give this: by the
        time a human reacts the policy has already sent several ticks at 21 Hz,
        and on the first run of a new runner -- possibly with glassware in the
        grippers -- those ticks are the entire risk the step exists to retire.
        """
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(
                Path(directory), extra_argv=("--chain.reset.only=true",)
            )

        self.assertEqual(run.exit_code, cli.EXIT_OK)
        ends = run.of("stage_end")
        self.assertEqual(len(ends), 11)
        self.assertEqual(
            [event["kind"] for event in ends],
            ["reset"] * 11,
            "reset-only must expand to ramps ONLY",
        )
        self.assertEqual(
            [event["terminator"] for event in ends], [TERMINATED_BY_REACHED] * 11
        )
        self.assertEqual(
            [event["stage_id"] for event in ends],
            ["reset_pre_01"] + [f"reset_to_{k:02d}" for k in range(2, 12)],
        )
        # Each ramp's dmax IS the boundary gap: nothing between two ramps moves
        # the arm, so ramp k+1 anchors exactly where ramp k left it. That is the
        # measurement bring-up (2) is for.
        for event in ends[1:]:
            detail = event["reason_detail"]
            self.assertGreater(detail["reset_dmax"], 0.0)
            self.assertLess(detail["reach_err"], detail["reset_tol_rad"])
        # Every ramp went through the state-only predict path, and SAID SO per
        # stage. The claim this carries is "the swap was installed and every tick
        # of every reset used it", which is what the 12.5 -> ~21 Hz fix depends
        # on; the SKIPPED CONVERSION itself is asserted in
        # test_chain_unit.ResetPredictPathTest (MockRobot has no cameras, so
        # there is no image key here to skip).
        for event in ends:
            detail = event["reason_detail"]
            self.assertEqual(
                detail["predict_path"],
                "state_only",
                f"{event['stage_id']}: {detail}",
            )
            self.assertEqual(
                detail["predict_calls"],
                event["frames"],
                f"{event['stage_id']}: one lightweight predict per recorded "
                "frame, or the swap was bypassed on some ticks",
            )
        # And it did not leak: after a full run upstream's function is back.
        from lerobot.scripts import lerobot_record
        from lerobot.utils import control_utils

        self.assertIs(
            lerobot_record.predict_action,
            control_utils.predict_action,
            "the reset swap must be undone -- a leak would run the NEXT "
            "lerobot-record's policy without its camera frames",
        )
        # No checkpoint was loaded.
        trial_start = run.of("trial_start")[0]
        self.assertEqual(
            trial_start["policies"],
            [],
            "a ramp loads no weights, so trial_start must report no policy",
        )
        self.assertEqual(run.trial_end["reason"], TRIAL_REASON_COMPLETED)

    # ---------------------------------------------------------------- preflight

    def test_a_chain_config_that_also_lists_stages_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            run_argv = [
                "--config_path",
                str(TESTS / "configs" / "smoke_mock.yaml"),
                "--version=2",
                "--chain.enabled=true",
                f"--chain.params_path={MOCK_PARAMS}",
                "--chain.model.policy_path=mock://progress",
                "--chain.model.has_progress=true",
                f"--dataset.root={temporary / 'dataset'}",
                f"--output.root={temporary / 'outputs'}",
            ]
            self.assertEqual(cli.main(run_argv), cli.EXIT_PREFLIGHT)

    def test_a_mock_chain_without_has_progress_is_refused(self) -> None:
        """With no checkpoint there is no action width to read, so it is required.

        The YAML is edited on disk rather than overridden on the command line:
        draccus cannot decode `--chain.model.has_progress=None` into
        `bool | None` ("Couldn't parse 'None' into a bool"), and a test that
        asserted the absence of a key by passing a string would be asserting
        draccus's parser, not this gate.
        """
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            unset = temporary / "chain_no_has_progress.yaml"
            unset.write_text(
                CHAIN_CONFIG.read_text(encoding="utf-8").replace(
                    "has_progress: true", "has_progress: null"
                ),
                encoding="utf-8",
            )
            argv = [
                "--config_path",
                str(unset),
                f"--chain.params_path={MOCK_PARAMS}",
                f"--dataset.root={temporary / 'dataset'}",
                f"--output.root={temporary / 'outputs'}",
            ]
            self.assertEqual(cli.main(argv), cli.EXIT_PREFLIGHT)

    def test_a_params_file_the_loader_rejects_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            broken = temporary / "broken.json"
            document = json.loads(MOCK_PARAMS.read_text())
            document["stages"].pop("5")
            broken.write_text(json.dumps(document))
            argv = [
                "--config_path",
                str(CHAIN_CONFIG),
                f"--chain.params_path={broken}",
                f"--dataset.root={temporary / 'dataset'}",
                f"--output.root={temporary / 'outputs'}",
            ]
            self.assertEqual(cli.main(argv), cli.EXIT_PREFLIGHT)

    def test_an_fps_mismatch_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            argv = [
                "--config_path",
                str(CHAIN_CONFIG),
                f"--chain.params_path={MOCK_PARAMS}",
                "--dataset.fps=30",
                f"--dataset.root={temporary / 'dataset'}",
                f"--output.root={temporary / 'outputs'}",
            ]
            self.assertEqual(cli.main(argv), cli.EXIT_PREFLIGHT)

    def test_a_mock_chain_with_a_one_hot_k_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            argv = [
                "--config_path",
                str(CHAIN_CONFIG),
                f"--chain.params_path={MOCK_PARAMS}",
                "--chain.model.onehot_k=11",
                f"--dataset.root={temporary / 'dataset'}",
                f"--output.root={temporary / 'outputs'}",
            ]
            # Exit 2, not a traceback: it is an operator error, and
            # check_chain_definitions names it. load_chain_bundles keeps its own
            # ValueError as a backstop for a caller that skips preflight.
            self.assertEqual(cli.main(argv), cli.EXIT_PREFLIGHT)

    def test_a_threshold_out_of_range_exits_two(self) -> None:
        """[중요]8, through the gate the runner actually runs.

        ``stall_s: 0`` leaves ``_covered_window`` asking for a zero-length window,
        which is never COVERED, so no stage could complete and all eleven would run
        to their timeouts -- a chain that "fails" on eleven stages that finished.
        """
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for override in (
                "--chain.completion.stall_s=0",
                "--chain.completion.departure_arm_rad=0",
                "--chain.completion.timeout_factor=1.0",
                "--chain.reset.tol_rad=0",
                "--chain.reset.ceiling_factor=1.0",
                "--chain.completion.stall_track_rad=0.01",
            ):
                with self.subTest(override):
                    argv = [
                        "--config_path",
                        str(CHAIN_CONFIG),
                        f"--chain.params_path={MOCK_PARAMS}",
                        override,
                        f"--dataset.root={temporary / 'dataset'}",
                        f"--output.root={temporary / 'outputs'}",
                    ]
                    self.assertEqual(
                        cli.main(argv),
                        cli.EXIT_PREFLIGHT,
                        "an out-of-range threshold must be refused before the "
                        "robot is energised, not silently turn a rule off",
                    )

    def test_a_non_finite_action_sends_a_hold_and_breaks_the_chain(self) -> None:
        """[중요]4, end to end: the gate is the last thing before the arms.

        ``mock://nan`` runs stage 1 normally for four ticks and then puts a NaN in
        one arm joint. What must be true afterwards: the chain STOPPED, the exit
        code is 4 and not 1 (the gate sets the same flags Esc does, and "a human
        stopped this" is the bucket a batch script treats as nobody's fault), and
        NOT ONE action that reached the robot carried a non-finite number.
        """
        import math

        sent: list[dict] = []
        original_make_robot = record_adapter.make_robot

        def capturing_make_robot(robot_config):
            robot = original_make_robot(robot_config)
            inner = robot.send_action

            def send_action(action):
                sent.append(dict(action))
                return inner(action)

            robot.send_action = send_action
            return robot

        record_adapter.make_robot = capturing_make_robot
        try:
            with tempfile.TemporaryDirectory() as directory:
                run = run_chain(Path(directory), policy_path="mock://nan", to_stage=2)
        finally:
            record_adapter.make_robot = original_make_robot

        self.assertEqual(
            run.exit_code,
            cli.EXIT_CHAIN_FAILED,
            "4 (the chain broke), not 1 (a human stopped it): the gate stops the "
            "run through the same flags Esc sets, so cli has to tell them apart",
        )
        self.assertTrue(sent, "the capture must have seen the run's actions")
        offenders = [
            (index, key, value)
            for index, action in enumerate(sent)
            for key, value in action.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ]
        self.assertEqual(
            offenders,
            [],
            "lerobot's clamp would have passed the NaN through to "
            "set_all_positions (robots/utils.py:99-104, "
            "widowxai_follower.py:272); the gate is what stops it",
        )
        # And the chain did not merely refuse to start: stage 1 ran first.
        ends = run.of("stage_end")
        self.assertEqual([event["stage_id"] for event in ends], ["reset_pre_01", "t01"])
        self.assertGreater(
            ends[1]["frames"], 0, "the stage was running when the NaN arrived"
        )

    def test_the_version_one_smoke_config_still_parses(self) -> None:
        """Version 1 is read unchanged; it is the regression test for the rest."""
        parsed = config.parse_config(
            ["--config_path", str(TESTS / "configs" / "smoke_mock.yaml")]
        )
        self.assertEqual(parsed.version, 1)
        self.assertFalse(parsed.chain.enabled)
        self.assertEqual([stage.id for stage in parsed.stages], ["s4", "s5"])
        for stage in parsed.stages:
            self.assertEqual(stage.kind, "policy")
            self.assertIsNone(stage.stage_number)
            self.assertEqual(stage.required_terminator, [])


def _events_with_timers(*presses: tuple[float, dict[str, bool]]):
    """A make_keyboard_events stand-in: core's three flags, flipped by timers.

    Each press is (delay_s, flags) -- the dict is updated in place, which is
    exactly what core's pynput callback does (control_utils.py:135-160).
    """

    def keyboard_events():
        events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
        for delay, flags in presses:
            threading.Timer(delay, lambda f=flags: events.update(f)).start()
        return None, events

    return keyboard_events


TELEOP_ARGV = ("--teleop.type=stage_runner_mock_teleop", "--teleop_time_s=10")


class TeleopPhaseTests(unittest.TestCase):
    """The leader-arm window before the first stage (runner._run_teleop_phase)."""

    def tearDown(self) -> None:  # noqa: N802
        leaked = [name for name in FORBIDDEN_MODULES if name in sys.modules]
        self.assertEqual(leaked, [], "the teleop window must not pull in the robot SDK")

    def test_no_teleop_config_means_no_teleop_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(Path(directory), to_stage=1)
        self.assertEqual(run.of("teleop_start"), [])
        self.assertEqual(run.of("teleop_end"), [])

    def test_right_arrow_ends_the_phase_and_the_chain_runs_unrecorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(
                Path(directory),
                to_stage=2,
                extra_argv=TELEOP_ARGV,
                patch_events=_events_with_timers((0.6, {"exit_early": True})),
            )
        starts, ends = run.of("teleop_start"), run.of("teleop_end")
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["ended_by"], "arrow", ends[0])
        self.assertLess(ends[0]["elapsed_s"], 5.0, "the arrow, not the 10 s ceiling, ended it")
        self.assertEqual(ends[0]["stop_base_path"], "primary")
        # Nothing of the teleop window went into the episode buffer.
        self.assertEqual(starts[0]["frame_idx"], 0)
        self.assertEqual(ends[0]["frame_idx"], 0)
        first_stage = run.of("stage_start")[0]
        self.assertEqual(first_stage["frame_idx"], 0)
        self.assertEqual(first_stage["stage_id"], "reset_pre_01")
        self.assertTrue(run.trial_end["completed"], run.trial_end)
        self.assertEqual(run.exit_code, 0)

    def test_left_arrow_restarts_the_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(
                Path(directory),
                to_stage=1,
                extra_argv=TELEOP_ARGV,
                patch_events=_events_with_timers(
                    (0.4, {"rerecord_episode": True, "exit_early": True}),
                    (1.0, {"exit_early": True}),
                ),
            )
        ends = run.of("teleop_end")
        self.assertEqual([e["ended_by"] for e in ends], ["left_arrow_restart", "arrow"], ends)
        self.assertEqual([e["attempt"] for e in ends], [1, 2])
        self.assertTrue(run.trial_end["completed"], run.trial_end)

    def test_timeout_ends_the_phase_without_a_keypress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(
                Path(directory),
                to_stage=1,
                extra_argv=("--teleop.type=stage_runner_mock_teleop", "--teleop_time_s=0.7"),
                patch_events=_events_with_timers(),
            )
        ends = run.of("teleop_end")
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["ended_by"], "timeout", ends[0])
        # The ceiling is an abort, never a start: no stage may have moved.
        self.assertFalse(run.trial_end["completed"], run.trial_end)
        self.assertEqual(run.of("stage_end")[0]["frames"], 0)

    def test_esc_in_the_phase_aborts_before_any_stage_moves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(
                Path(directory),
                to_stage=2,
                extra_argv=TELEOP_ARGV,
                patch_events=_events_with_timers((0.4, {"stop_recording": True, "exit_early": True})),
            )
        ends = run.of("teleop_end")
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["ended_by"], "esc", ends[0])
        self.assertLess(ends[0]["elapsed_s"], 5.0, "ESC must end the window at once, not at the ceiling")
        self.assertFalse(run.trial_end["completed"], run.trial_end)
        # The abort lands on the first stage before it ran a single tick.
        stage_ends = run.of("stage_end")
        self.assertTrue(stage_ends, "the first stage must still emit its (aborted) stage_end")
        self.assertEqual(stage_ends[0]["frames"], 0, stage_ends[0])
        self.assertNotEqual(run.exit_code, 0)

    def test_esc_pressed_before_the_window_skips_it_entirely(self) -> None:
        # ESC (or SIGHUP through the latch) during connect: the window must not run
        # at all -- a leader dragging the followers for the whole ceiling, then a
        # tidy `esc`, is the case the review found.
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(
                Path(directory),
                to_stage=2,
                extra_argv=("--teleop.type=stage_runner_mock_teleop", "--teleop_time_s=3"),
                patch_events=_events_with_timers((0.0, {"stop_recording": True, "exit_early": True})),
            )
        self.assertEqual(run.of("teleop_start"), [], "no loop may have been entered")
        ends = run.of("teleop_end")
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["ended_by"], "esc_before_start", ends[0])
        self.assertEqual(ends[0]["elapsed_s"], 0.0)
        self.assertFalse(run.trial_end["completed"], run.trial_end)
        self.assertEqual(run.of("stage_end")[0]["frames"], 0)

    def test_right_arrow_is_ignored_while_the_arm_is_outside_the_initial_gate(self) -> None:
        # pose_rad=2.0 parks every joint 2 rad away from the mock stage-1 pose: the arrow
        # must bounce (arrow_not_ready), the window continues, and the ceiling then aborts.
        with tempfile.TemporaryDirectory() as directory:
            run = run_chain(
                Path(directory),
                to_stage=1,
                extra_argv=(
                    "--teleop.type=stage_runner_mock_teleop",
                    "--teleop.pose_rad=2.0",
                    "--teleop_time_s=1.6",
                ),
                patch_events=_events_with_timers((0.5, {"exit_early": True})),
            )
        ends = run.of("teleop_end")
        self.assertEqual([e["ended_by"] for e in ends], ["arrow_not_ready", "timeout"], ends)
        self.assertGreater(ends[0]["gap_rad"], ends[0]["gate_rad"])
        self.assertTrue(ends[0]["gap_joint"])
        self.assertEqual(run.of("teleop_start")[-1]["attempt"], 2)
        self.assertFalse(run.trial_end["completed"], run.trial_end)
        self.assertEqual(run.of("stage_end")[0]["frames"], 0, "nothing may have moved")

    def test_base_velocity_from_the_leader_is_zeroed_by_default(self) -> None:
        from lerobot.processor.core import TransitionKey

        from stage_runner.record_adapter import TeleopBaseZeroStep

        step = TeleopBaseZeroStep()
        transition = {TransitionKey.ACTION: {"left_joint_0.pos": 0.2, "x.vel": 0.3, "theta.vel": -0.1}}
        out = step(transition)
        self.assertEqual(out[TransitionKey.ACTION]["x.vel"], 0.0)
        self.assertEqual(out[TransitionKey.ACTION]["theta.vel"], 0.0)
        self.assertEqual(out[TransitionKey.ACTION]["left_joint_0.pos"], 0.2)
        self.assertEqual(transition[TransitionKey.ACTION]["x.vel"], 0.3, "the input is not mutated")
        self.assertIs(step({TransitionKey.ACTION: {"a.pos": 1.0}})[TransitionKey.ACTION].get("x.vel"), None)
        self.assertEqual(step.transform_features({"k": 1}), {"k": 1})


if __name__ == "__main__":
    unittest.main()

