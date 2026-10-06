"""Unit tests for the chain: arithmetic, windows, expansion, vocabulary.

``unittest``, not pytest. pytest is NOT installed in this repo's venv (measured
2026-10-06), and installing it is out of scope for a session that must not touch
the lock file, so the new tests have to run on the stdlib runner::

    STAGE_RUNNER_NO_PLUGINS=1 PYTHONPATH=packages/stage_runner/src \\
      .venv/bin/python -m unittest discover -s packages/stage_runner/tests \\
      -p 'test_chain*.py' -v

The inherited ``test_smoke_runner.py`` / ``test_aggregate.py`` are pytest-style
bare functions taking ``tmp_path``; ``unittest discover`` collects ZERO tests
from them, silently. ``-p 'test_chain*.py'`` makes that explicit rather than
accidental, and the shim that runs the inherited ones is in the session's
scratchpad.

NO ROBOT SDK. ``STAGE_RUNNER_NO_PLUGINS=1`` stops ``register_plugins()`` from
importing ``lerobot_robot_trossen`` (and through it ``trossen_slate``), and
:meth:`NoRobotSdkMixin.tearDown` asserts neither module ever reached
``sys.modules``. The plugin file that IS tested here
(``task_onehot_patch.py``) is loaded from its PATH with stubs standing in for
the package, never by importing the package.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("STAGE_RUNNER_NO_PLUGINS", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_SRC = REPO_ROOT / "packages" / "stage_runner" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

MOCK_PARAMS = Path(__file__).resolve().parent / "data" / "stage_params_mock.json"
PLUGIN_SRC = (
    REPO_ROOT
    / "packages"
    / "lerobot_robot_trossen"
    / "src"
    / "lerobot_robot_trossen"
)

from stage_runner import chain_params as cp  # noqa: E402
from stage_runner import completion as comp  # noqa: E402
from stage_runner import reset_policy as rp  # noqa: E402
from stage_runner.events import EventLog  # noqa: E402
from stage_runner.mock_robot import REALISTIC_JOINT_NAMES  # noqa: E402
from stage_runner.results import (  # noqa: E402
    EMITTED_TERMINATOR_VALUES,
    TERMINATED_BY_COMPLETE,
    TERMINATED_BY_MANUAL,
    TERMINATED_BY_NOT_REACHED,
    TERMINATED_BY_REACHED,
    TERMINATED_BY_VALUES,
)

FORBIDDEN_MODULES = ("lerobot_robot_trossen", "trossen_slate")


class NoRobotSdkMixin:
    """Every chain test asserts the Trossen SDK was never imported.

    The repo rule is that code importing a robot SDK is not executed off the
    robot PC. The SDK arrives through exactly two doors --
    ``register_plugins()``'s third-party scan and any direct
    ``import lerobot_robot_trossen`` -- and this closes both by checking after
    the fact rather than by trusting a comment.
    """

    def tearDown(self) -> None:  # noqa: N802 - unittest's name
        leaked = [name for name in FORBIDDEN_MODULES if name in sys.modules]
        assert not leaked, (
            f"robot SDK module(s) {leaked} were imported by this test. The chain "
            "must be testable with no robot: use STAGE_RUNNER_NO_PLUGINS=1 and "
            "load plugin files by path."
        )


def load_params():
    return cp.load_chain_params(MOCK_PARAMS, expected_fps=21)


# --------------------------------------------------------------------------- #
# stage_params.json loader
# --------------------------------------------------------------------------- #


class ChainParamsTest(NoRobotSdkMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(MOCK_PARAMS.read_text(encoding="utf-8"))

    def test_the_mock_file_loads_and_poses_are_keyed_by_name(self) -> None:
        params = load_params()
        self.assertEqual(sorted(params.stages), list(range(1, 12)))
        self.assertEqual(params.fps, 21)
        stage = params.stage(4)
        # By NAME, not by index: this is the whole point of arm_joint_order.
        self.assertEqual(set(stage.start_pose), set(cp.ARM_JOINT_NAMES))
        self.assertEqual(len(stage.start_pose), 12)
        self.assertTrue(stage.end_poses)
        for pose in stage.end_poses:
            self.assertEqual(set(pose), set(cp.ARM_JOINT_NAMES))

    def test_timeout_is_p90_times_the_factor(self) -> None:
        stage = load_params().stage(1)
        self.assertAlmostEqual(stage.timeout_s(1.3), stage.p90_s * 1.3)

    def test_a_reordered_arm_joint_order_still_maps_by_name(self) -> None:
        """The ORDER is free, the NAMES are not -- and the zip must follow it."""
        document = json.loads(MOCK_PARAMS.read_text(encoding="utf-8"))
        order = list(document["arm_joint_order"])
        reversed_order = list(reversed(order))
        document["arm_joint_order"] = reversed_order
        for stage in document["stages"].values():
            stage["start_pose_rad_arm12"] = list(
                reversed(stage["start_pose_rad_arm12"])
            )
            stage["end_poses_rad_arm12"] = [
                list(reversed(pose)) for pose in stage["end_poses_rad_arm12"]
            ]
        reordered = cp.parse_chain_params(document, expected_fps=21)
        straight = load_params()
        for number in range(1, 12):
            self.assertEqual(
                reordered.stage(number).start_pose,
                straight.stage(number).start_pose,
                "a file written in another joint order must produce the SAME "
                "name -> radians mapping; this is the data-layout trap "
                "(real 16-D is [left7, right7, x, theta], MuJoCo teleop puts "
                "the base first) the name keying exists to close",
            )

    def _reject(self, mutate, needle: str) -> None:
        document = json.loads(MOCK_PARAMS.read_text(encoding="utf-8"))
        mutate(document)
        with self.assertRaises(cp.ChainParamsError) as caught:
            cp.parse_chain_params(document, expected_fps=21)
        self.assertIn(needle, str(caught.exception))

    def test_rejects_a_future_schema_version(self) -> None:
        self._reject(lambda d: d.update(schema_version=2), "schema_version")

    def test_rejects_an_fps_that_disagrees_with_the_run(self) -> None:
        document = json.loads(MOCK_PARAMS.read_text(encoding="utf-8"))
        with self.assertRaises(cp.ChainParamsError) as caught:
            cp.parse_chain_params(document, expected_fps=30)
        self.assertIn("dataset.fps=30", str(caught.exception))

    def test_rejects_a_missing_stage(self) -> None:
        self._reject(lambda d: d["stages"].pop("7"), "missing stage(s) ['7']")

    def test_rejects_an_unexpected_stage_key(self) -> None:
        self._reject(
            lambda d: d["stages"].__setitem__("12", d["stages"]["11"]),
            "unexpected key(s) ['12']",
        )

    def test_rejects_a_wrong_joint_name(self) -> None:
        def mutate(document):
            document["arm_joint_order"][0] = "left_joint_9"

        self._reject(mutate, "is not the 12 arm joints")

    def test_rejects_eleven_joints(self) -> None:
        def mutate(document):
            document["arm_joint_order"].pop()

        self._reject(mutate, "expected 12 joint names")

    def test_rejects_a_non_finite_pose_value(self) -> None:
        def mutate(document):
            # JSON has no NaN literal, so the parsed document carries the string
            # -- which is exactly how a hand-edited file would break.
            document["stages"]["3"]["start_pose_rad_arm12"][5] = "NaN"

        self._reject(mutate, "not a finite number")

    def test_rejects_a_bool_in_a_pose(self) -> None:
        """``float(True) == 1.0``: a bool would become a 1-radian command."""

        def mutate(document):
            document["stages"]["2"]["start_pose_rad_arm12"][0] = True

        self._reject(mutate, "not a finite number")

    def test_rejects_p10_equal_to_p90(self) -> None:
        def mutate(document):
            document["stages"]["5"]["p10_s"] = document["stages"]["5"]["p90_s"]
            document["stages"]["5"]["p50_s"] = document["stages"]["5"]["p90_s"]

        self._reject(mutate, "strictly less than p90_s")

    def test_rejects_an_out_of_order_quantile_triple(self) -> None:
        def mutate(document):
            document["stages"]["6"]["p50_s"] = 99.0

        self._reject(mutate, "p10_s <= p50_s <= p90_s")

    def test_rejects_empty_end_poses(self) -> None:
        def mutate(document):
            document["stages"]["8"]["end_poses_rad_arm12"] = []

        self._reject(mutate, "non-empty list of poses")

    def test_rejects_a_non_positive_tolerance(self) -> None:
        def mutate(document):
            document["stages"]["9"]["end_pose_tol_rad"] = 0.0

        self._reject(mutate, "end_pose_tol_rad")

    def test_collects_every_problem_in_one_message(self) -> None:
        def mutate(document):
            document["stages"]["1"]["p10_s"] = -1.0
            document["stages"]["2"]["end_pose_tol_rad"] = -0.1
            document["stages"].pop("3")

        document = json.loads(MOCK_PARAMS.read_text(encoding="utf-8"))
        mutate(document)
        with self.assertRaises(cp.ChainParamsError) as caught:
            cp.parse_chain_params(document, expected_fps=21)
        message = str(caught.exception)
        self.assertIn("p10_s", message)
        self.assertIn("end_pose_tol_rad", message)
        self.assertIn("missing stage(s)", message)

    def test_start_pose_deg_pairs_are_left_then_right(self) -> None:
        params = load_params()
        left, right = params.start_pose_deg_pairs(2)
        self.assertEqual((len(left), len(right)), (6, 6))
        self.assertAlmostEqual(
            left[0], math.degrees(params.stage(2).start_pose["left_joint_0"])
        )
        self.assertAlmostEqual(
            right[5], math.degrees(params.stage(2).start_pose["right_joint_5"])
        )


# --------------------------------------------------------------------------- #
# minimum-jerk ramp
# --------------------------------------------------------------------------- #


class MinimumJerkTest(NoRobotSdkMixin, unittest.TestCase):
    def test_endpoints_are_exact(self) -> None:
        self.assertEqual(rp.quintic(0.0), 0.0)
        self.assertEqual(rp.quintic(1.0), 1.0)
        # Clamped outside [0, 1] rather than extrapolating: the hold phase keeps
        # calling with u > 1.
        self.assertEqual(rp.quintic(1.7), 1.0)
        self.assertEqual(rp.quintic(-0.3), 0.0)

    def test_boundary_velocity_is_zero(self) -> None:
        """f'(0) = f'(1) = 0, so the ramp neither jumps nor overshoots."""
        step = 1e-6
        self.assertLess(abs(rp.quintic(step) - rp.quintic(0.0)) / step, 1e-5)
        self.assertLess(abs(rp.quintic(1.0) - rp.quintic(1.0 - step)) / step, 1e-5)

    def test_peak_slope_is_fifteen_eighths(self) -> None:
        slopes = []
        samples = 20001
        for index in range(samples - 1):
            u0 = index / (samples - 1)
            u1 = (index + 1) / (samples - 1)
            slopes.append((rp.quintic(u1) - rp.quintic(u0)) / (u1 - u0))
        self.assertAlmostEqual(max(slopes), rp.QUINTIC_PEAK_SLOPE, places=4)

    def test_tick_step_of_the_design_case_is_under_the_clamp(self) -> None:
        """Delta = 0.7 rad, T = 1.5 s, 21 Hz -> about 0.041 rad/tick <= 0.07.

        0.07 is 70% of ``max_relative_target`` (0.1 rad), i.e. the margin the
        design asks for. N = ceil(1.5 * 21) = 32, so the bound is
        1.875 * 0.7 / 32 = 0.0410 -- the plan's "0.042" was computed against
        N = 31.5 before the ceiling.
        """
        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        target = dict(anchor)
        target["left_joint_1"] = 0.7
        plan = rp.plan_reset(
            stage_number=2,
            anchor=anchor,
            target=target,
            fps=21,
            settings=rp.ResetSettings(t_min_s=1.5, v_des_rad_s=0.524),
        )
        self.assertAlmostEqual(plan.delta_max, 0.7)
        self.assertAlmostEqual(plan.duration_s, 1.5)
        self.assertEqual(plan.ticks, 32)
        self.assertLessEqual(plan.max_step_rad, 0.07)
        self.assertAlmostEqual(plan.max_step_rad, 1.875 * 0.7 / 32, places=6)

        # And the realised per-tick steps never exceed the bound.
        steps = [
            abs(rp.quintic((t + 1) / plan.ticks) - rp.quintic(t / plan.ticks)) * 0.7
            for t in range(plan.ticks)
        ]
        self.assertLessEqual(max(steps), plan.max_step_rad + 1e-12)
        self.assertLessEqual(max(steps), 0.07)

    def test_the_worst_measured_boundary_also_fits_under_the_clamp(self) -> None:
        """1.28 rad is the largest of the ten measured stage boundaries (§91)."""
        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        target = dict(anchor, left_joint_3=1.28)
        plan = rp.plan_reset(
            stage_number=4,
            anchor=anchor,
            target=target,
            fps=21,
            settings=rp.ResetSettings(),
        )
        self.assertAlmostEqual(plan.duration_s, 1.28 / 0.524, places=6)
        self.assertLess(plan.max_step_rad, 0.1, "the 0.1 rad clamp must not fire")

    def test_duration_floors_at_t_min(self) -> None:
        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        plan = rp.plan_reset(
            stage_number=1,
            anchor=anchor,
            target=dict(anchor, left_joint_0=0.01),
            fps=21,
            settings=rp.ResetSettings(),
        )
        self.assertAlmostEqual(plan.duration_s, 1.5)

    def test_refuses_a_jump_over_the_threshold(self) -> None:
        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        with self.assertRaises(rp.ResetAbort) as caught:
            rp.plan_reset(
                stage_number=3,
                anchor=anchor,
                target=dict(anchor, right_joint_2=1.6),
                fps=21,
                settings=rp.ResetSettings(max_jump_rad=1.5),
            )
        self.assertIn("nothing moved", str(caught.exception))

    def test_refuses_a_non_finite_target(self) -> None:
        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        with self.assertRaises(rp.ResetAbort):
            rp.plan_reset(
                stage_number=3,
                anchor=anchor,
                target=dict(anchor, right_joint_2=float("nan")),
                fps=21,
                settings=rp.ResetSettings(),
            )

    def test_refuses_a_missing_joint(self) -> None:
        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        target = dict(anchor)
        target.pop("left_joint_4")
        with self.assertRaises(rp.ResetAbort) as caught:
            rp.plan_reset(
                stage_number=3,
                anchor=anchor,
                target=target,
                fps=21,
                settings=rp.ResetSettings(),
            )
        self.assertIn("left_joint_4", str(caught.exception))


class ResetPolicyTest(NoRobotSdkMixin, unittest.TestCase):
    """The ramp as the loop drives it: a fake batch in, an action vector out."""

    # 14 arm positions then the two base velocities: MobileAIRobot's own order
    # (action_features = arms.action_features | {x.vel, theta.vel}).
    ACTION_NAMES = tuple(
        f"{name}.pos" for name in REALISTIC_JOINT_NAMES
    ) + ("x.vel", "theta.vel")

    def _policy(self, *, progress: bool = False, fps: int = 21):
        import torch

        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        target = {name: 0.3 for name in cp.ARM_JOINT_NAMES}
        settings = rp.ResetSettings(t_min_s=0.5, settle_s=0.1, tol_rad=0.05)
        plan = rp.plan_reset(
            stage_number=5,
            anchor=anchor,
            target=target,
            fps=fps,
            settings=settings,
        )
        action_names = list(self.ACTION_NAMES)
        if progress:
            action_names.append(comp.PROGRESS_KEY)
        state_names = [f"{name}.pos" for name in _realistic_names()]
        events: dict[str, bool] = {"exit_early": False}
        policy = rp.ResetPolicy(
            plan,
            events,
            action_names=action_names,
            state_names=state_names,
            settings=settings,
        )
        return policy, plan, events, action_names, state_names, torch

    def test_the_ramp_lands_exactly_on_the_target_and_holds(self) -> None:
        policy, plan, events, action_names, state_names, torch = self._policy()
        measured = {name: 0.0 for name in state_names}
        last = None
        for _ in range(plan.ticks):
            batch = {
                "observation.state": torch.tensor(
                    [[measured[name] for name in state_names]], dtype=torch.float32
                )
            }
            action = policy.select_action(batch).squeeze(0).tolist()
            commanded = dict(zip(action_names, action))
            # Position-controlled: the mock arm ends up where it was told.
            for name in state_names:
                measured[name] = commanded[name]
            last = commanded
        for name in cp.ARM_JOINT_NAMES:
            # places=6, not 9: the action crosses a float32 tensor on its way
            # out (make_robot_action squeezes and casts), so 0.3 comes back as
            # 0.30000001192. That is the REAL path's precision -- the claim under
            # test is that the ramp commands the target and not 0.999 of it,
            # which a quintic that did not reach f(1)=1 exactly would fail by
            # three orders of magnitude more than this.
            self.assertAlmostEqual(
                last[f"{name}.pos"],
                0.3,
                places=6,
                msg="the last commanded value must be the target; a 0.999 "
                "landing leaves a bias at every one of eleven boundaries",
            )
        # And in exact arithmetic the final fraction is 1.0, not 0.999...
        self.assertEqual(rp.quintic(plan.ticks / plan.ticks), 1.0)

    def test_the_base_is_commanded_to_zero_every_tick(self) -> None:
        policy, plan, events, action_names, state_names, torch = self._policy()
        batch = {
            "observation.state": torch.zeros(
                (1, len(state_names)), dtype=torch.float32
            )
        }
        for _ in range(3):
            action = dict(zip(action_names, policy.select_action(batch).squeeze(0).tolist()))
            self.assertEqual(action["x.vel"], 0.0)
            self.assertEqual(action["theta.vel"], 0.0)

    def test_the_gripper_is_held_at_its_measured_value_not_zeroed(self) -> None:
        policy, plan, events, action_names, state_names, torch = self._policy()
        measured = {name: 0.0 for name in state_names}
        # A gripper squeezing an object sits at a nonzero carriage position.
        measured["left_left_carriage_joint.pos"] = 0.021
        measured["right_left_carriage_joint.pos"] = 0.019
        batch = {
            "observation.state": torch.tensor(
                [[measured[name] for name in state_names]], dtype=torch.float32
            )
        }
        action = dict(zip(action_names, policy.select_action(batch).squeeze(0).tolist()))
        self.assertAlmostEqual(action["left_left_carriage_joint.pos"], 0.021, places=6)
        self.assertAlmostEqual(action["right_left_carriage_joint.pos"], 0.019, places=6)

    def test_progress_is_zero_during_a_reset(self) -> None:
        policy, plan, events, action_names, state_names, torch = self._policy(
            progress=True
        )
        batch = {
            "observation.state": torch.zeros(
                (1, len(state_names)), dtype=torch.float32
            )
        }
        action = dict(zip(action_names, policy.select_action(batch).squeeze(0).tolist()))
        self.assertEqual(action[comp.PROGRESS_KEY], 0.0)

    def test_arrival_needs_settle_ticks_and_sets_exit_early(self) -> None:
        policy, plan, events, action_names, state_names, torch = self._policy()
        measured = {name: 0.0 for name in state_names}
        ticks = 0
        while not events["exit_early"] and ticks < plan.ticks + 200:
            batch = {
                "observation.state": torch.tensor(
                    [[measured[name] for name in state_names]], dtype=torch.float32
                )
            }
            action = dict(
                zip(action_names, policy.select_action(batch).squeeze(0).tolist())
            )
            for name in state_names:
                measured[name] = action[name]
            ticks += 1
        self.assertTrue(policy.reached)
        self.assertTrue(events["exit_early"])
        # plan.ticks + settle_ticks - 1: the tick on which the ramp first
        # commands the target (tick == plan.ticks, u == 1) is also the first
        # tick on which arrival can be COUNTED, because the measurement that
        # tick carries is already within tol of the target -- the previous
        # command was 0.9888 of the way there. So the settle window's first
        # sample is that tick, not the one after it.
        self.assertGreaterEqual(
            ticks,
            plan.ticks + plan.settle_ticks - 1,
            "arrival must not be declared before the ramp has commanded the "
            "target AND held it for settle_ticks",
        )
        self.assertLess(policy.worst_error_rad, 0.05)

    def test_an_arm_that_never_follows_is_never_reached(self) -> None:
        policy, plan, events, action_names, state_names, torch = self._policy()
        # The arm is stuck at the anchor: every command is ignored.
        batch = {
            "observation.state": torch.zeros(
                (1, len(state_names)), dtype=torch.float32
            )
        }
        for _ in range(plan.ticks + 100):
            policy.select_action(batch)
        self.assertFalse(policy.reached)
        self.assertFalse(events["exit_early"])
        self.assertAlmostEqual(policy.worst_error_rad, 0.3, places=6)

    def test_wait_until_still_accepts_a_still_arm_and_rejects_a_moving_one(self) -> None:
        class Arm:
            def __init__(self, drift: float) -> None:
                self.drift = drift
                self.reads = 0

            def get_observation(self):
                self.reads += 1
                return {
                    f"{name}.pos": self.drift * self.reads
                    for name in cp.ARM_JOINT_NAMES
                }

        settings = rp.ResetSettings(settle_check_tries=3, settle_check_tol_rad=0.02)
        reading, still = rp.wait_until_still(
            Arm(0.0), settings=settings, sleep=lambda _s: None
        )
        self.assertTrue(still)
        self.assertEqual(len(reading), 12)

        reading, still = rp.wait_until_still(
            Arm(0.5), settings=settings, sleep=lambda _s: None
        )
        self.assertFalse(still, "a drifting arm must not be used as a ramp anchor")


def _realistic_names():
    return REALISTIC_JOINT_NAMES


# --------------------------------------------------------------------------- #
# completion monitor -- the four synthetic sequences
# --------------------------------------------------------------------------- #


class CompletionMonitorTest(NoRobotSdkMixin, unittest.TestCase):
    """Synthetic tick sequences against an injected clock.

    The clock is injected so a loaded machine cannot change the answer: these
    are assertions about the RULE, and the rule is defined in seconds.
    """

    def setUp(self) -> None:
        self.params = load_params().stage(1)  # p10 0.3, p50 0.5, p90 0.8
        self.names = cp.ARM_JOINT_NAMES
        self.now = 0.0
        self.events: dict[str, bool] = {"exit_early": False}
        self.settings = comp.CompletionSettings(
            p_done=0.95, p_hold_s=0.2, stall_s=0.4, stall_arm_rad=0.05, stall_base=0.05
        )

    def _monitor(self, *, has_progress: bool):
        monitor = comp.CompletionMonitorStep(
            self.events,
            settings=self.settings,
            arm_joint_names=self.names,
            clock=lambda: self.now,
        )
        monitor.begin_stage(self.params, has_progress=has_progress, started=0.0)
        return monitor

    def _tick(self, monitor, *, progress=None, offset=0.0, base=0.0, dt=1 / 21.0):
        from lerobot.processor import create_transition

        self.now += dt
        action = {
            f"{name}.pos": self.params.start_pose[name] + offset for name in self.names
        }
        observation = dict(action)
        action["x.vel"] = base
        action["theta.vel"] = 0.0
        if progress is not None:
            action[comp.PROGRESS_KEY] = progress
        monitor(create_transition(action=action, observation=observation))

    def test_a_progress_ramp_that_stalls_completes(self) -> None:
        monitor = self._monitor(has_progress=True)
        for index in range(60):
            self._tick(monitor, progress=min(1.0, index / 4.0))
            if self.events["exit_early"]:
                break
        self.assertTrue(self.events["exit_early"])
        result = monitor.end_stage()
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, comp.REASON_PROGRESS)
        self.assertGreaterEqual(result.elapsed_s, self.params.p10_s)
        self.assertGreaterEqual(result.stall_s, self.settings.stall_s - 1 / 21.0)
        self.assertEqual(result.p_last, 1.0)

    def test_stalled_but_progress_only_point_three_is_not_complete(self) -> None:
        monitor = self._monitor(has_progress=True)
        for _ in range(80):
            self._tick(monitor, progress=0.3)
        self.assertFalse(self.events["exit_early"])
        self.assertIsNone(monitor.end_stage())

    def test_a_two_second_mid_task_pause_then_motion_is_not_complete(self) -> None:
        """23-35% of the demonstrations of t02/06/08 pause for over 2 s (§93).

        Three phases, and the stage must only complete at the end of the third:
        a two-second pause mid-task, a resumption that travels, and the real
        stop.
        """
        monitor = self._monitor(has_progress=True)

        # 1. Pause: nothing moves for 2 s, well past stall_s (0.4) and p10
        #    (0.3). Stall and the elapsed floor are both satisfied; p is not.
        for _ in range(42):
            self._tick(monitor, progress=0.6)
        self.assertFalse(
            self.events["exit_early"],
            "a mid-task pause must not complete the stage: p is 0.6, not >= 0.95",
        )
        self.assertGreater(self.now, 2.0)

        # 2. It resumes and travels, with p now above the threshold. The arm's
        #    peak-to-peak over the window is what refuses here -- |cmd - meas|
        #    is zero on every tick, because the mock arm follows perfectly.
        offset = 0.0
        for _ in range(42):
            offset += 0.01
            self._tick(monitor, progress=1.0, offset=offset)
            if self.events["exit_early"]:
                break
        self.assertFalse(
            self.events["exit_early"],
            "while the arm is still travelling the stage is not complete, even "
            "with p at 1.0",
        )

        # 3. It really stops. NOW it completes.
        for _ in range(42):
            self._tick(monitor, progress=1.0, offset=offset)
            if self.events["exit_early"]:
                break
        result = monitor.end_stage()
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, comp.REASON_PROGRESS)

    def test_a_completion_signal_before_p10_waits(self) -> None:
        monitor = self._monitor(has_progress=True)
        # p is 1.0 and nothing moves from the very first tick, but p10 is 0.3 s
        # and stall_s is 0.4 s, so the earliest honest completion is at 0.4 s.
        fired_at = None
        for _ in range(40):
            self._tick(monitor, progress=1.0)
            if self.events["exit_early"]:
                fired_at = self.now
                break
        self.assertIsNotNone(fired_at)
        self.assertGreaterEqual(fired_at, self.params.p10_s)
        self.assertGreaterEqual(fired_at, self.settings.stall_s)

    def test_a_commanded_base_velocity_blocks_completion(self) -> None:
        monitor = self._monitor(has_progress=True)
        for _ in range(60):
            self._tick(monitor, progress=1.0, base=0.2)
        self.assertFalse(
            self.events["exit_early"],
            "a base still being driven is not a stage that has finished",
        )

    def test_a_tracked_slow_ramp_blocks_completion(self) -> None:
        """|cmd - meas| stays tiny while the arm crosses the workspace."""
        monitor = self._monitor(has_progress=True)
        for index in range(80):
            self._tick(monitor, progress=1.0, offset=0.01 * index)
        self.assertFalse(
            self.events["exit_early"],
            "peak-to-peak over the window is what catches a perfectly tracked "
            "steady ramp; |cmd - meas| alone cannot",
        )

    def test_a_frozen_loop_does_not_trivially_stall(self) -> None:
        monitor = self._monitor(has_progress=True)
        self._tick(monitor, progress=1.0, dt=1 / 21.0)
        # The loop blocks for 5 s (a RealSense async_read), then one tick.
        self._tick(monitor, progress=1.0, dt=5.0)
        self.assertFalse(
            self.events["exit_early"],
            "two samples spanning a 5 s gap are not a covered stall window",
        )

    def test_a_sixteen_d_model_completes_by_stall_plus_end_pose(self) -> None:
        monitor = self._monitor(has_progress=False)
        # end_poses[0] of the mock file is start_pose + 0.05, tol 0.12.
        for _ in range(40):
            self._tick(monitor, offset=0.05)
            if self.events["exit_early"]:
                break
        self.assertTrue(self.events["exit_early"])
        result = monitor.end_stage()
        self.assertEqual(result.reason, comp.REASON_STALL_NN)
        self.assertEqual(result.nn_index, 0)
        self.assertLess(result.nn_dist, self.params.end_pose_tol_rad)
        self.assertIsNone(result.p_last)

    def test_a_sixteen_d_model_stalled_far_from_any_end_pose_is_not_complete(
        self,
    ) -> None:
        monitor = self._monitor(has_progress=False)
        for _ in range(60):
            self._tick(monitor, offset=0.9)
        self.assertFalse(self.events["exit_early"])
        self.assertIsNone(monitor.end_stage())

    def test_the_window_is_cleared_between_stages(self) -> None:
        """record_loop never resets the robot action pipeline; begin_stage does."""
        monitor = self._monitor(has_progress=True)
        for _ in range(40):
            self._tick(monitor, progress=1.0)
            if self.events["exit_early"]:
                break
        self.assertTrue(self.events["exit_early"])
        monitor.end_stage()
        self.events["exit_early"] = False

        monitor.begin_stage(self.params, has_progress=True, started=self.now)
        self._tick(monitor, progress=1.0)
        self.assertFalse(
            self.events["exit_early"],
            "the previous stage's stalled window must not complete this stage "
            "on its first tick",
        )

    def test_a_missing_progress_key_makes_the_monitor_blind_not_wrong(self) -> None:
        monitor = self._monitor(has_progress=True)
        for _ in range(60):
            self._tick(monitor, progress=None)  # 17-D declared, key absent
        self.assertFalse(
            self.events["exit_early"],
            "a monitor that cannot read its signal must never declare completion",
        )
        self.assertIsNone(monitor.end_stage())

    def test_the_monitor_never_mutates_the_action(self) -> None:
        from lerobot.processor import create_transition

        monitor = self._monitor(has_progress=True)
        action = {f"{name}.pos": 0.0 for name in self.names}
        action.update({"x.vel": 0.0, "theta.vel": 0.0, comp.PROGRESS_KEY: 1.0})
        snapshot = dict(action)
        observation = {f"{name}.pos": 0.0 for name in self.names}
        self.now += 1 / 21.0
        returned = monitor(create_transition(action=action, observation=observation))
        self.assertEqual(action, snapshot)
        from lerobot.processor.core import TransitionKey

        self.assertIs(
            returned[TransitionKey.ACTION],
            action,
            "the action dict is the SAME object record_loop hands to "
            "build_dataset_frame; popping `progress` out of it would KeyError "
            "on the dataset write one line later",
        )

    def test_an_exception_in_the_monitor_does_not_escape(self) -> None:
        monitor = self._monitor(has_progress=True)
        # A transition shaped nothing like the real one.
        self.assertIsNotNone(monitor({}))
        self.assertFalse(self.events["exit_early"])
        self.assertIsNone(monitor.end_stage())


# --------------------------------------------------------------------------- #
# chain expansion, 17-D declaration, event vocabulary
# --------------------------------------------------------------------------- #


class ChainExpansionTest(NoRobotSdkMixin, unittest.TestCase):
    def _config(self, **chain_overrides):
        from stage_runner import config as cfg_module
        from stage_runner.mock_robot import MockRobotConfig

        chain = cfg_module.ChainConfig(
            enabled=True,
            params_path=str(MOCK_PARAMS),
            from_stage=chain_overrides.pop("from_stage", 1),
            to_stage=chain_overrides.pop("to_stage", 11),
            model=cfg_module.ChainModelConfig(
                policy_path="local/ckpt",
                onehot_k=chain_overrides.pop("onehot_k", 11),
            ),
            completion=cfg_module.CompletionConfig(
                allow_manual_complete=chain_overrides.pop(
                    "allow_manual_complete", True
                )
            ),
            reset=cfg_module.ResetConfig(initial=chain_overrides.pop("initial", True)),
        )
        assert not chain_overrides, chain_overrides
        return cfg_module.StageRunnerConfig(
            robot=MockRobotConfig(), version=2, chain=chain
        ), cfg_module

    def test_the_full_chain_is_eleven_policy_stages_and_eleven_resets(self) -> None:
        config, cfg_module = self._config()
        stages = cfg_module.expand_chain(config, load_params())
        self.assertEqual(len(stages), 22)
        policies = [s for s in stages if s.kind == cfg_module.STAGE_KIND_POLICY]
        resets = [s for s in stages if s.kind == cfg_module.STAGE_KIND_RESET]
        self.assertEqual(len(policies), 11)
        self.assertEqual(len(resets), 11)
        self.assertEqual(sum(1 for s in resets if s.initial_reset), 1)
        self.assertEqual(
            sum(1 for s in resets if not s.initial_reset),
            10,
            "ten of the eleven resets sit at a boundary between two policy "
            "stages; the design brief's '10 resets' means those",
        )

    def test_the_order_alternates_reset_then_policy(self) -> None:
        config, cfg_module = self._config()
        stages = cfg_module.expand_chain(config, load_params())
        self.assertEqual(
            [s.id for s in stages[:5]],
            ["reset_pre_01", "t01", "reset_to_02", "t02", "reset_to_03"],
        )
        self.assertEqual(stages[-1].id, "t11")

    def test_the_one_hot_index_is_one_based_and_equals_the_stage_number(self) -> None:
        config, cfg_module = self._config()
        stages = cfg_module.expand_chain(config, load_params())
        for stage in stages:
            if stage.kind == cfg_module.STAGE_KIND_POLICY:
                self.assertEqual(stage.onehot_index, stage.stage_number)
            else:
                self.assertIsNone(stage.onehot_index)

    def test_no_one_hot_model_gets_no_index(self) -> None:
        config, cfg_module = self._config(onehot_k=None)
        stages = cfg_module.expand_chain(config, load_params())
        self.assertTrue(
            all(s.onehot_index is None for s in stages),
            "a checkpoint with no one-hot step must get no index, or the "
            "executor raises rather than running the wrong conditioning",
        )

    def test_stage_index_is_not_the_stage_number(self) -> None:
        config, cfg_module = self._config()
        stages = cfg_module.expand_chain(config, load_params())
        eleventh = next(s for s in stages if s.stage_number == 11 and s.kind == "policy")
        self.assertEqual(stages.index(eleventh), 21)

    def test_required_terminators(self) -> None:
        config, cfg_module = self._config()
        stages = cfg_module.expand_chain(config, load_params())
        for stage in stages:
            if stage.kind == cfg_module.STAGE_KIND_POLICY:
                self.assertEqual(
                    stage.required_terminator,
                    [TERMINATED_BY_COMPLETE, TERMINATED_BY_MANUAL],
                )
            else:
                self.assertEqual(stage.required_terminator, [TERMINATED_BY_REACHED])

    def test_manual_completion_off_removes_it_from_the_requirement(self) -> None:
        config, cfg_module = self._config(allow_manual_complete=False)
        stages = cfg_module.expand_chain(config, load_params())
        policy = next(s for s in stages if s.kind == cfg_module.STAGE_KIND_POLICY)
        self.assertEqual(policy.required_terminator, [TERMINATED_BY_COMPLETE])

    def test_initial_false_drops_the_first_reset(self) -> None:
        config, cfg_module = self._config(initial=False)
        stages = cfg_module.expand_chain(config, load_params())
        self.assertEqual(len(stages), 21)
        self.assertEqual(stages[0].id, "t01")

    def test_a_sub_range_runs_only_those_stages(self) -> None:
        config, cfg_module = self._config(from_stage=4, to_stage=6)
        stages = cfg_module.expand_chain(config, load_params())
        self.assertEqual(
            [s.id for s in stages],
            ["reset_pre_04", "t04", "reset_to_05", "t05", "reset_to_06", "t06"],
        )

    def test_timeouts_come_from_p90_times_the_factor(self) -> None:
        config, cfg_module = self._config()
        params = load_params()
        stages = cfg_module.expand_chain(config, params)
        policy = next(s for s in stages if s.id == "t03")
        self.assertAlmostEqual(
            policy.terminator.timeout_s, params.stage(3).p90_s * 1.3
        )

    def test_reset_stages_declare_no_checkpoint(self) -> None:
        config, cfg_module = self._config()
        stages = cfg_module.expand_chain(config, load_params())
        for stage in stages:
            if stage.kind == cfg_module.STAGE_KIND_RESET:
                self.assertEqual(stage.policy_path, "")
                self.assertEqual(stage.executor, cfg_module.EXECUTOR_CHAIN_RESET)
                self.assertEqual(stage.instruction, f"reset:{stage.stage_number:02d}")


class ResetOnlyExpansionTest(NoRobotSdkMixin, unittest.TestCase):
    """``chain.reset.only`` drops every policy stage. Bring-up steps (2) and (3)."""

    def _stages(self, **overrides):
        from stage_runner import config as cfg_module
        from stage_runner.mock_robot import MockRobotConfig

        chain = cfg_module.ChainConfig(
            enabled=True,
            params_path=str(MOCK_PARAMS),
            from_stage=overrides.pop("from_stage", 1),
            to_stage=overrides.pop("to_stage", 11),
            model=cfg_module.ChainModelConfig(policy_path="local/ckpt", onehot_k=11),
            reset=cfg_module.ResetConfig(
                only=True, initial=overrides.pop("initial", False)
            ),
        )
        assert not overrides, overrides
        config = cfg_module.StageRunnerConfig(
            robot=MockRobotConfig(), version=2, chain=chain
        )
        return cfg_module.expand_chain(config, load_params()), cfg_module

    def test_reset_only_is_eleven_ramps_and_no_policy_stage(self) -> None:
        stages, cfg_module = self._stages()
        self.assertEqual(len(stages), 11)
        self.assertTrue(
            all(s.kind == cfg_module.STAGE_KIND_RESET for s in stages),
            "a policy stage here would drive the arms in the one mode whose "
            "whole purpose is that nothing does",
        )
        self.assertEqual([s.stage_number for s in stages], list(range(1, 12)))

    def test_reset_only_forces_the_first_ramp_in(self) -> None:
        """Even with `initial: false` -- otherwise the first gap is not a boundary."""
        stages, _ = self._stages(initial=False)
        self.assertEqual(stages[0].id, "reset_pre_01")
        self.assertTrue(stages[0].initial_reset)

    def test_reset_only_keeps_the_same_ids_as_the_full_chain(self) -> None:
        """Same executor, same plan, same ids: the ramps are not a second path."""
        reset_only, cfg_module = self._stages()
        from stage_runner.mock_robot import MockRobotConfig

        full = cfg_module.expand_chain(
            cfg_module.StageRunnerConfig(
                robot=MockRobotConfig(),
                version=2,
                chain=cfg_module.ChainConfig(
                    enabled=True,
                    params_path=str(MOCK_PARAMS),
                    model=cfg_module.ChainModelConfig(
                        policy_path="local/ckpt", onehot_k=11
                    ),
                ),
            ),
            load_params(),
        )
        self.assertEqual(
            [s.id for s in reset_only],
            [s.id for s in full if s.kind == cfg_module.STAGE_KIND_RESET],
        )

    def test_reset_only_over_a_sub_range(self) -> None:
        stages, _ = self._stages(from_stage=7, to_stage=9)
        self.assertEqual(
            [s.id for s in stages],
            ["reset_pre_07", "reset_to_08", "reset_to_09"],
        )


class LateBoundRecordLoopTest(NoRobotSdkMixin, unittest.TestCase):
    """``call_record_loop`` must resolve ``record_loop`` ON THE MODULE, per call.

    Three fork plugins rebind ``lerobot_record.record_loop`` from
    ``register_plugins()``, which runs AFTER ``record_adapter`` is imported. An
    import-time ``from ... import record_loop`` would keep calling the original,
    and the failure is invisible: no phase tag, so base_serial_rearm refuses for
    the whole run, basevel.csv's `phase` column is empty and pose_guide prints
    nothing -- while the chain appears to work.
    """

    def test_a_rebound_record_loop_is_the_one_that_gets_called(self) -> None:
        from lerobot.scripts import lerobot_record

        from stage_runner import record_adapter
        from stage_runner.policies import PolicyBundle

        calls: list[dict] = []

        def sentinel(**kwargs):
            calls.append(kwargs)

        original = lerobot_record.record_loop
        lerobot_record.record_loop = sentinel
        try:
            record_adapter.call_record_loop(
                robot=object(),
                events={},
                fps=21,
                processors=record_adapter.make_processors(),
                dataset=object(),
                bundle=PolicyBundle(
                    stage_id="t01",
                    policy_path="mock://hold",
                    config=object(),
                    policy=object(),
                    preprocessor=object(),
                    postprocessor=object(),
                ),
                control_time_s=1.0,
                single_task="t",
            )
        finally:
            lerobot_record.record_loop = original

        self.assertEqual(
            len(calls),
            1,
            "the function rebound AFTER record_adapter was imported must be the "
            "one that runs -- that is what the fork's loop_rate_log, "
            "chunk_execution_patch and base_serial_rearm all depend on",
        )
        # And `dataset` by KEYWORD: @safe_stop_image_writer reads
        # kwargs.get("dataset") (image_writer.py:26-38).
        self.assertIn("dataset", calls[0])

    def test_record_adapter_does_not_hold_its_own_binding(self) -> None:
        from stage_runner import record_adapter

        self.assertFalse(
            hasattr(record_adapter, "record_loop"),
            "a module-level `record_loop` name here means `from ... import` came "
            "back; the plugins rebind the attribute on lerobot_record, not here",
        )


class ExtraActionFeatureTest(NoRobotSdkMixin, unittest.TestCase):
    def _robot(self):
        from stage_runner.mock_robot import MockRobot, MockRobotConfig

        return MockRobot(
            MockRobotConfig(
                joint_count=14,
                realistic_joint_names=True,
                include_base_in_state=False,
            )
        )

    def test_no_extra_names_reproduces_the_pre_chain_features(self) -> None:
        from stage_runner import record_adapter

        robot = self._robot()
        processors = record_adapter.make_processors()
        baseline = record_adapter.build_dataset_features(robot, processors, False)
        again = record_adapter.build_dataset_features(robot, processors, False, ())
        self.assertEqual(
            baseline,
            again,
            "extra_action_names=() must be byte-for-byte the old behaviour; "
            "M1 (16-D) depends on it",
        )
        from lerobot.utils.constants import ACTION

        self.assertEqual(len(baseline[ACTION]["names"]), 16)

    def test_progress_widens_names_and_shape_together(self) -> None:
        from lerobot.utils.constants import ACTION

        from stage_runner import record_adapter

        robot = self._robot()
        processors = record_adapter.make_processors()
        features = record_adapter.build_dataset_features(
            robot, processors, False, (comp.PROGRESS_KEY,)
        )
        names = features[ACTION]["names"]
        self.assertEqual(len(names), 17)
        self.assertEqual(names[-1], comp.PROGRESS_KEY)
        self.assertEqual(
            tuple(features[ACTION]["shape"]),
            (17,),
            "the SHAPE is what dataset_to_policy_features reads into the "
            "PolicyFeature make_policy assigns, so a name list that grew "
            "without the shape would leave the ACT head at 16",
        )

    def test_make_robot_action_produces_the_progress_key(self) -> None:
        import torch
        from lerobot.policies.utils import make_robot_action

        from stage_runner import record_adapter

        robot = self._robot()
        processors = record_adapter.make_processors()
        features = record_adapter.build_dataset_features(
            robot, processors, False, (comp.PROGRESS_KEY,)
        )
        tensor = torch.arange(17, dtype=torch.float32).unsqueeze(0)
        action = make_robot_action(tensor, features)
        self.assertIn(comp.PROGRESS_KEY, action)
        self.assertEqual(action[comp.PROGRESS_KEY], 16.0)

    def test_the_robot_ignores_the_progress_key(self) -> None:
        """MockRobot reproduces MobileAIRobot's arm filter and base `.get`."""
        robot = self._robot()
        robot.connect()
        action = {f"{name}.pos": 0.1 for name in _realistic_names()}
        action.update({"x.vel": 0.0, "theta.vel": 0.0, comp.PROGRESS_KEY: 0.77})
        echo = robot.send_action(action)
        self.assertNotIn(comp.PROGRESS_KEY, echo)
        robot.disconnect()

    def test_a_colliding_extra_name_is_refused(self) -> None:
        from stage_runner import record_adapter

        robot = self._robot()
        processors = record_adapter.make_processors()
        with self.assertRaises(ValueError):
            record_adapter.build_dataset_features(
                robot, processors, False, ("x.vel",)
            )

    def test_extra_action_names_maps_widths(self) -> None:
        from stage_runner import preflight

        self.assertEqual(preflight.extra_action_names(16, 16), ())
        self.assertEqual(preflight.extra_action_names(17, 16), ("progress",))
        self.assertEqual(preflight.extra_action_names(None, 16), ())
        with self.assertRaises(preflight.PreflightError):
            preflight.extra_action_names(18, 16)
        with self.assertRaises(preflight.PreflightError):
            preflight.extra_action_names(14, 16)


class TerminatorVocabularyTest(NoRobotSdkMixin, unittest.TestCase):
    def test_the_chain_outcomes_are_in_the_outcome_vocabulary(self) -> None:
        for value in (
            TERMINATED_BY_COMPLETE,
            TERMINATED_BY_REACHED,
            TERMINATED_BY_NOT_REACHED,
        ):
            self.assertIn(value, TERMINATED_BY_VALUES)

    def test_completion_is_planned_only(self) -> None:
        self.assertNotIn(
            "completion",
            TERMINATED_BY_VALUES,
            "`completion` is a RULE a stage ends by, not an outcome. In the "
            "outcome vocabulary it would become a bucket the aggregator counts",
        )
        self.assertIn("completion", EMITTED_TERMINATOR_VALUES)

    def test_the_event_log_accepts_every_chain_terminator(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with EventLog(path, "unit") as log:
                for value in EMITTED_TERMINATOR_VALUES:
                    log.emit("stage_end", stage_id="t01", terminator=value)
                with self.assertRaises(ValueError):
                    log.emit("stage_end", stage_id="t01", terminator="finished")
            lines = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        self.assertEqual(
            [line["terminator"] for line in lines], list(EMITTED_TERMINATOR_VALUES)
        )

    def test_the_aggregator_tolerates_an_unknown_terminator(self) -> None:
        """It warns and buckets; only the WRITER refuses. Both are deliberate."""
        from stage_runner import aggregate

        self.assertEqual(
            aggregate.TERMINATED_BY_VALUES,
            TERMINATED_BY_VALUES,
            "the aggregator must bucket on the same outcome vocabulary the "
            "runner writes, or a chain outcome becomes an unnamed bucket",
        )


# --------------------------------------------------------------------------- #
# the one-hot plugin, loaded by PATH so the robot SDK is never imported
# --------------------------------------------------------------------------- #


def _load_task_onehot_module():
    """Load ``task_onehot_patch.py`` from its path, with the package stubbed.

    ``import lerobot_robot_trossen.task_onehot_patch`` would execute that
    package's ``__init__``, which imports ``mobileai`` and through it
    ``trossen_slate`` -- the robot SDK. ``eval_najy.sh:117-121`` already reads
    this file the same way (``spec_from_file_location`` on the path) for exactly
    this reason, and this is that convention in Python.

    The module itself imports only ``torch`` and
    ``lerobot.processor.pipeline``, so no stub is needed for it; the stub below
    exists so the PARENT package name resolves without being executed.
    """
    package = types.ModuleType("lerobot_robot_trossen")
    package.__path__ = [str(PLUGIN_SRC)]
    sys.modules.setdefault("lerobot_robot_trossen_stub", package)
    spec = importlib.util.spec_from_file_location(
        "task_onehot_patch_under_test", PLUGIN_SRC / "task_onehot_patch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TaskOneHotStepTest(unittest.TestCase):
    """``set_stage`` must re-point the index AND re-announce.

    No NoRobotSdkMixin here: the stub module registered under a different name
    is the point, and the real package must still be absent -- asserted
    directly below.
    """

    def test_set_stage_repoints_and_reannounces(self) -> None:
        import torch
        from lerobot.processor import create_transition
        from lerobot.processor.core import TransitionKey

        module = _load_task_onehot_module()
        self.assertNotIn("trossen_slate", sys.modules)
        self.assertNotIn("lerobot_robot_trossen", sys.modules)

        step = module.TaskOneHotStep(stage=1, k=11, expected_dim=27)
        state = torch.zeros((1, 14), dtype=torch.float32)

        def hot_index(step):
            transition = create_transition(
                observation={"observation.state": state.clone()}
            )
            out = step(transition)
            widened = out[TransitionKey.OBSERVATION]["observation.state"]
            self.assertEqual(widened.shape[-1], 27)
            return int(torch.argmax(widened[0, 16:]).item())

        self.assertEqual(hot_index(step), 0)
        self.assertTrue(step._announced, "the first tick announces")

        step.set_stage(7)
        self.assertEqual(step.index, 6)
        self.assertFalse(
            step._announced,
            "set_stage must clear _announced: the 'stage i/K active' line is "
            "the only evidence in the log that the one-hot switched, and a "
            "chain that announced stage 1 and silently ran 2..11 would be "
            "indistinguishable from one that never switched",
        )
        self.assertEqual(hot_index(step), 6)

    def test_reset_does_not_undo_set_stage(self) -> None:
        module = _load_task_onehot_module()
        step = module.TaskOneHotStep(stage=1, k=11, expected_dim=27)
        step.set_stage(5)
        step.reset()  # record_loop resets the preprocessor on entry to EVERY stage
        self.assertEqual(
            step.index,
            4,
            "record_loop resets the preprocessor on loop ENTRY, so a reset() "
            "that re-pointed the index would undo the caller's choice right "
            "after it was made",
        )

    def test_set_stage_refuses_an_out_of_range_stage(self) -> None:
        module = _load_task_onehot_module()
        step = module.TaskOneHotStep(stage=1, k=11, expected_dim=27)
        for bad in (0, 12, -1):
            with self.assertRaises(RuntimeError):
                step.set_stage(bad)

    def test_insert_task_onehot_returns_the_step(self) -> None:
        module = _load_task_onehot_module()
        from lerobot.processor.normalize_processor import NormalizerProcessorStep

        class Preprocessor:
            def __init__(self) -> None:
                self.steps = [object.__new__(NormalizerProcessorStep)]

        class Feature:
            shape = (27,)

        class Config:
            input_features = {"observation.state": Feature()}

        pre = Preprocessor()
        step = module.insert_task_onehot(pre, Config(), 3, 11)
        self.assertIsInstance(step, module.TaskOneHotStep)
        self.assertIs(pre.steps[0], step, "it must land BEFORE the normalizer")
        self.assertEqual(step.index, 2)

    def test_insert_refuses_a_checkpoint_of_the_wrong_width(self) -> None:
        module = _load_task_onehot_module()

        class Feature:
            shape = (16,)

        class Config:
            input_features = {"observation.state": Feature()}

        with self.assertRaises(RuntimeError):
            module.insert_task_onehot(object(), Config(), 1, 11)


if __name__ == "__main__":
    unittest.main()
