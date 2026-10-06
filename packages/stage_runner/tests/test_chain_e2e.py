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
        for event in reset_ends:
            self.assertEqual(
                event["terminator"],
                TERMINATED_BY_REACHED,
                f"reset {event['stage_id']} did not arrive: {event['reason']}",
            )
            detail = event["reason_detail"]
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


if __name__ == "__main__":
    unittest.main()
