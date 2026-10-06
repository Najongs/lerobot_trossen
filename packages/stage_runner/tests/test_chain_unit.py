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


class AnchorDriftTest(NoRobotSdkMixin, unittest.TestCase):
    """M-1: the mid-entry backstop is a FIRST-TICK check, not a per-tick one."""

    ACTION_NAMES = ResetPolicyTest.ACTION_NAMES

    def _policy(self, *, delta: float, max_jump_rad: float, fps: int = 21):
        import torch

        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        target = {name: delta for name in cp.ARM_JOINT_NAMES}
        settings = rp.ResetSettings(
            t_min_s=0.5, settle_s=0.1, tol_rad=0.05, max_jump_rad=max_jump_rad
        )
        plan = rp.plan_reset(
            stage_number=6,
            anchor=anchor,
            target=target,
            fps=fps,
            settings=settings,
        )
        state_names = [f"{name}.pos" for name in REALISTIC_JOINT_NAMES]
        events: dict[str, bool] = {"exit_early": False}
        policy = rp.ResetPolicy(
            plan,
            events,
            action_names=list(self.ACTION_NAMES),
            state_names=state_names,
            settings=settings,
        )
        return policy, plan, events, state_names, torch

    def _batch(self, torch, state_names, measured):
        return {
            "observation.state": torch.tensor(
                [[measured[name] for name in state_names]], dtype=torch.float32
            )
        }

    def test_the_drift_check_runs_exactly_once_over_a_whole_ramp(self) -> None:
        """It is called before the FIRST command and never again.

        `_anchor_drift` grows monotonically as the ramp does its job -- by the
        last tick it IS delta_max -- so a per-tick check measures the ramp's own
        progress. Counting the calls is what pins the fix: the behaviour it
        caused (a refusal at the END of a correct ramp) is only reachable when
        the drift limit is below delta_max, which plan_reset refuses outright.
        """
        policy, plan, events, state_names, torch = self._policy(
            delta=1.0, max_jump_rad=1.1
        )
        calls = {"n": 0}
        original = policy._anchor_drift

        def counted(measured):
            calls["n"] += 1
            return original(measured)

        policy._anchor_drift = counted

        measured = {name: 0.0 for name in state_names}
        ticks = 0
        while not events["exit_early"] and ticks < plan.ticks + 200:
            action = dict(
                zip(
                    self.ACTION_NAMES,
                    policy.select_action(
                        self._batch(torch, state_names, measured)
                    ).squeeze(0).tolist(),
                )
            )
            for name in state_names:
                measured[name] = action[name]
            ticks += 1

        self.assertEqual(
            calls["n"],
            1,
            "the anchor-drift backstop must run on the first tick only; it ran "
            f"{calls['n']} times over a {plan.ticks}-tick ramp",
        )
        self.assertEqual(policy.refused, "")
        self.assertTrue(policy.reached)

    def test_a_mid_ramp_measurement_past_the_limit_does_not_refuse(self) -> None:
        """The behaviour the per-tick check produced, now impossible.

        One tick reports the arm 1.2 rad from the anchor -- past the 1.1 rad
        limit this ramp was planned under. With a per-tick check that is a
        refusal, the policy holds position and the reset reports `not_reached`,
        which is a chain failure. It is not the first tick, so it is not the
        thing the backstop exists for.
        """
        policy, plan, events, state_names, torch = self._policy(
            delta=1.0, max_jump_rad=1.1
        )
        measured = {name: 0.0 for name in state_names}
        policy.select_action(self._batch(torch, state_names, measured))
        self.assertEqual(policy.refused, "")

        spiked = {name: 1.2 for name in state_names}
        policy.select_action(self._batch(torch, state_names, spiked))
        self.assertEqual(
            policy.refused,
            "",
            "a measurement past the limit on a tick other than the first is not "
            "what this backstop judges -- it is the ramp, or noise",
        )
        self.assertFalse(events["exit_early"])

    def test_the_first_tick_past_the_limit_still_refuses(self) -> None:
        """The backstop itself is intact: the arm moved before the ramp began."""
        policy, plan, events, state_names, torch = self._policy(
            delta=1.0, max_jump_rad=1.1
        )
        moved = {name: 1.3 for name in state_names}
        values = policy.select_action(
            self._batch(torch, state_names, moved)
        ).squeeze(0).tolist()
        self.assertIn("moved", policy.refused)
        self.assertTrue(events["exit_early"])
        # And the refusal action HOLDS the measurement, it does not ramp.
        action = dict(zip(self.ACTION_NAMES, values))
        for name in cp.ARM_JOINT_NAMES:
            self.assertAlmostEqual(action[f"{name}.pos"], 1.3, places=5)


class ArrivalWindowTest(NoRobotSdkMixin, unittest.TestCase):
    """M-3: arrival is a RATIO over the settle window, not a consecutive run."""

    ACTION_NAMES = ResetPolicyTest.ACTION_NAMES

    def _policy(self, fps: int = 21):
        import torch

        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        target = {name: 0.3 for name in cp.ARM_JOINT_NAMES}
        # settle_s 1.0 at 21 Hz = 21 ticks, the production window.
        settings = rp.ResetSettings(t_min_s=0.5, settle_s=1.0, tol_rad=0.05)
        plan = rp.plan_reset(
            stage_number=3,
            anchor=anchor,
            target=target,
            fps=fps,
            settings=settings,
        )
        state_names = [f"{name}.pos" for name in REALISTIC_JOINT_NAMES]
        events: dict[str, bool] = {"exit_early": False}
        policy = rp.ResetPolicy(
            plan,
            events,
            action_names=list(self.ACTION_NAMES),
            state_names=state_names,
            settings=settings,
        )
        return policy, plan, events, state_names, settings, torch

    def _drive(self, *, glitches: set[int]):
        """Ramp with the arm following, glitching the MEASUREMENT on given ticks.

        Returns ``(policy, plan, ticks spent)``. A glitch is one tick reporting
        the arm 0.2 rad off target: an encoder read, or a pacing tick that
        landed outside tol. The command is unaffected.
        """
        policy, plan, events, state_names, settings, torch = self._policy()
        measured = {name: 0.0 for name in state_names}
        ticks = 0
        while not events["exit_early"] and ticks < plan.ticks + 200:
            reported = dict(measured)
            if ticks in glitches:
                for name in cp.ARM_JOINT_NAMES:
                    reported[f"{name}.pos"] = measured[f"{name}.pos"] + 0.2
            batch = {
                "observation.state": torch.tensor(
                    [[reported[name] for name in state_names]], dtype=torch.float32
                )
            }
            action = dict(
                zip(self.ACTION_NAMES, policy.select_action(batch).squeeze(0).tolist())
            )
            for name in state_names:
                measured[name] = action[name]
            ticks += 1
        return policy, plan, ticks

    def _shape(self):
        """``plan.ticks`` / ``plan.settle_ticks`` without driving anything."""
        _, plan, _, _, _, _ = self._policy()
        return plan

    def test_one_glitched_tick_in_twenty_one_still_arrives(self) -> None:
        shape = self._shape()
        self.assertEqual(shape.settle_ticks, 21, "the production settle window")
        # The 5th tick of the SETTLE window. Ticks inside the ramp do not count
        # toward arrival at all, so a glitch there would prove nothing.
        policy, plan, ticks = self._drive(glitches={shape.ticks + 4})
        self.assertTrue(policy.reached)
        self.assertLessEqual(
            ticks,
            plan.ticks + plan.settle_ticks,
            "a single tick outside tol must not restart the settle window: with "
            "a consecutive counter this needed 21 MORE ticks, and `not_reached` "
            "is a chain failure with no retry",
        )

    def test_three_glitched_ticks_in_twenty_one_delay_arrival(self) -> None:
        """90% is a threshold, not a free pass: 18 of 21 is not enough."""
        shape = self._shape()
        policy, plan, ticks = self._drive(
            glitches={shape.ticks + 2, shape.ticks + 6, shape.ticks + 10}
        )
        self.assertTrue(policy.reached, "it must still arrive, just later")
        self.assertGreater(
            ticks,
            plan.ticks + plan.settle_ticks,
            "three of twenty-one outside tol is below the 90% the window asks "
            "for, so arrival waits until they have rolled out of it",
        )

    # ------------------------------------------------- [중요]6: the newest sample

    def _feed(self, policy, torch, state_names, measured, *, glitch=0.0):
        """One tick. ``glitch`` offsets the REPORTED arm without moving it.

        The measurement then follows the command, like ``_drive`` does, so the
        glitch is one bad read and not a displacement that persists.
        """
        reported = dict(measured)
        if glitch:
            for name in cp.ARM_JOINT_NAMES:
                reported[f"{name}.pos"] = measured[f"{name}.pos"] + glitch
        batch = {
            "observation.state": torch.tensor(
                [[reported[name] for name in state_names]], dtype=torch.float32
            )
        }
        action = dict(
            zip(self.ACTION_NAMES, policy.select_action(batch).squeeze(0).tolist())
        )
        for name in state_names:
            measured[name] = action[name]

    def test_the_newest_sample_must_be_inside_tol(self) -> None:
        """A ratio is a statement about the window and says nothing about NOW.

        19 of 21 inside tol with the misses at the END is an arm that WAS settled
        and has started to drift, and ``reached`` is the signal that releases it
        to the next stage's policy. This drives the window to exactly the point
        where the ratio is satisfied and the newest answer is not.
        """
        policy, plan, events, state_names, settings, torch = self._policy()
        measured = {name: 0.0 for name in state_names}

        # 1. The ramp, with the arm following. Arrival answers only begin once the
        #    trajectory has commanded the target, so this leaves exactly one.
        for _ in range(plan.ticks):
            self._feed(policy, torch, state_names, measured)
        self.assertEqual(len(policy._arrival_window), 1)
        self.assertFalse(policy.reached)

        # 2. Fill the window to one short of full, all inside tol.
        for _ in range(plan.settle_ticks - 2):
            self._feed(policy, torch, state_names, measured)
        self.assertEqual(len(policy._arrival_window), plan.settle_ticks - 1)
        self.assertFalse(policy.reached, "the window must be FULL before arrival")

        # 3. The tick that fills it is OUTSIDE tol. 20 of 21 is 95%, past the 90%
        #    the ratio asks for -- so the ratio alone would declare arrival here.
        self._feed(policy, torch, state_names, measured, glitch=0.2)
        window = list(policy._arrival_window)
        self.assertEqual(len(window), plan.settle_ticks)
        self.assertEqual(
            sum(1 for value in window if value),
            plan.settle_ticks - 1,
            "20 of 21 inside tol: the RATIO is satisfied",
        )
        self.assertFalse(
            window[-1], "and the newest answer is the one outside it"
        )
        self.assertFalse(
            policy.reached,
            "an arm whose latest reading is outside tol has not arrived, however "
            "good the rest of the window was -- the order the ratio throws away "
            "is exactly what tells 'it has settled' from 'it is leaving'",
        )
        self.assertFalse(events["exit_early"])

        # 4. One good tick and it arrives: the delay is at most one tick.
        self._feed(policy, torch, state_names, measured)
        self.assertTrue(policy.reached)
        self.assertTrue(events["exit_early"])

    def test_an_arm_that_never_arrives_is_still_never_reached(self) -> None:
        """The ratio loosened the window; it did not remove the requirement."""
        policy, plan, events, state_names, settings, torch = self._policy()
        batch = {
            "observation.state": torch.zeros(
                (1, len(state_names)), dtype=torch.float32
            )
        }
        for _ in range(plan.ticks + 100):
            policy.select_action(batch)
        self.assertFalse(policy.reached)
        self.assertFalse(events["exit_early"])


class InitialResetJumpTest(NoRobotSdkMixin, unittest.TestCase):
    """C-2: the initial ramp (from a pose a HUMAN chose) has its own limit."""

    def _plan(self, *, gap: float, initial: bool, settings=None):
        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        target = dict(anchor, left_joint_2=gap)
        return rp.plan_reset(
            stage_number=1,
            anchor=anchor,
            target=target,
            fps=21,
            settings=settings or rp.ResetSettings(),
            initial=initial,
        )

    def test_an_initial_gap_over_point_six_is_refused(self) -> None:
        with self.assertRaises(rp.ResetAbort) as caught:
            self._plan(gap=0.8, initial=True)
        message = str(caught.exception)
        self.assertIn("nothing moved", message)
        self.assertIn("initial_max_jump_rad", message)
        self.assertIn(
            "0.3 rad",
            message,
            "the refusal has to say what to DO -- put the arms near the stage's "
            "start pose by hand -- or the operator raises the threshold instead",
        )

    def test_the_same_gap_is_accepted_at_a_boundary(self) -> None:
        """0.8 rad is INSIDE the measured 0.25-1.28 rad boundary spread."""
        plan = self._plan(gap=0.8, initial=False)
        self.assertAlmostEqual(plan.delta_max, 0.8)
        self.assertEqual(plan.jump_limit_rad, 1.5)
        self.assertFalse(plan.initial)

    def test_an_initial_gap_under_the_limit_is_planned_normally(self) -> None:
        plan = self._plan(gap=0.5, initial=True)
        self.assertAlmostEqual(plan.delta_max, 0.5)
        self.assertEqual(plan.jump_limit_rad, 0.6)
        self.assertTrue(plan.initial)
        self.assertEqual(plan.as_detail()["reset_initial"], True)

    def test_the_backstop_uses_the_limit_the_plan_was_approved_under(self) -> None:
        """Not `settings.max_jump_rad`: an initial ramp is held to 0.6 there too."""
        import torch

        plan = self._plan(gap=0.5, initial=True)
        state_names = [f"{name}.pos" for name in REALISTIC_JOINT_NAMES]
        events: dict[str, bool] = {"exit_early": False}
        policy = rp.ResetPolicy(
            plan,
            events,
            action_names=list(ResetPolicyTest.ACTION_NAMES),
            state_names=state_names,
            settings=rp.ResetSettings(),
        )
        # 0.7 rad from the anchor on the FIRST tick: inside max_jump_rad (1.5)
        # but outside the 0.6 this ramp was approved under.
        moved = {name: 0.7 for name in state_names}
        batch = {
            "observation.state": torch.tensor(
                [[moved[name] for name in state_names]], dtype=torch.float32
            )
        }
        policy.select_action(batch)
        self.assertIn("0.600", policy.refused)


def _realistic_names():
    return REALISTIC_JOINT_NAMES


# --------------------------------------------------------------------------- #
# completion monitor -- the four synthetic sequences
# --------------------------------------------------------------------------- #


class _FakePlan:
    """The two fields ``_classify_chain_policy`` reads off a stage plan."""

    def __init__(self, *, control_time_s: float) -> None:
        self.control_time_s = control_time_s


class _FakeContext:
    """Just enough StageContext for the classifier: no flags, fps 21."""

    def __init__(self) -> None:
        import types

        self.events: dict[str, bool] = {}
        self.config = types.SimpleNamespace(
            dataset=types.SimpleNamespace(fps=21)
        )


class _MonitorHarness(NoRobotSdkMixin):
    """Synthetic tick sequences against an injected clock.

    The clock is injected so a loaded machine cannot change the answer: these
    are assertions about the RULE, and the rule is defined in seconds.

    A mixin rather than a base TestCase, so the two suites below share the
    harness without ``unittest discover`` collecting every case twice.
    """

    # Every tick of a test that is NOT about the departure latch is sent with
    # the arm this far from where the stage STARTED, so the latch is satisfied
    # and the rule under test is the only thing that can refuse. 0.2 rad is twice
    # `departure_arm_rad`. The latch itself is tested in DepartureLatchTest, where
    # the offset is the variable.
    #
    # Sending this offset on every tick is NOT by itself a departure: the origin
    # is the stage's FIRST MEASURED pose (completion.py `_update_departure`), so a
    # constant offset is an arm that has been standing in one place since tick 1.
    # :meth:`_depart` is what actually moves it.
    DEPARTED = 0.2

    def _depart(self, monitor, *, has_progress: bool) -> None:
        """The minimum prelude that latches the departure, and nothing else.

        One tick at offset 0 -- which fixes the origin at the designated start
        pose, the ordinary case after a boundary reset -- then
        ``DEPARTURE_ARM_TICKS`` ticks at ``DEPARTED``, which is the consecutive
        run the latch now requires. ``progress`` is 0.0 for a 17-D monitor:
        below ``p_done``, so these four ticks can never complete anything
        themselves.
        """
        value = 0.0 if has_progress else None
        self._tick(monitor, progress=value, offset=0.0)
        for _ in range(comp.DEPARTURE_ARM_TICKS):
            self._tick(monitor, progress=value, offset=self.DEPARTED)

    def setUp(self) -> None:
        self.params = load_params().stage(1)  # p10 0.3, p50 0.5, p90 0.8
        self.names = cp.ARM_JOINT_NAMES
        self.now = 0.0
        self.events: dict[str, bool] = {"exit_early": False}
        self.settings = comp.CompletionSettings(
            p_done=0.95,
            p_hold_s=0.2,
            stall_s=0.4,
            stall_track_rad=0.08,
            stall_arm_rad=0.05,
            stall_base=0.05,
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

    def _tick(
        self,
        monitor,
        *,
        progress=None,
        offset=0.0,
        base=0.0,
        theta=0.0,
        track=0.0,
        dt=1 / 21.0,
    ):
        """One transition. ``offset`` moves BOTH command and measurement.

        ``track`` separates them: the measurement lags the command by that much,
        which is a tracking error and not motion. ``offset`` is motion.
        """
        from lerobot.processor import create_transition

        self.now += dt
        action = {
            f"{name}.pos": self.params.start_pose[name] + offset for name in self.names
        }
        observation = {key: value - track for key, value in action.items()}
        action["x.vel"] = base
        action["theta.vel"] = theta
        if progress is not None:
            action[comp.PROGRESS_KEY] = progress
        monitor(create_transition(action=action, observation=observation))


class CompletionMonitorTest(_MonitorHarness, unittest.TestCase):
    """The conjunction's four synthetic sequences, departure held satisfied.

    ``_monitor`` runs :meth:`_MonitorHarness._depart` here, so every test in this
    class starts with the latch already set and the rule it names is the only
    thing that can refuse. Before 2026-10-06 the premise was free -- a constant
    ``offset=DEPARTED`` was read as a departure -- and the negative tests in this
    class would have passed on a monitor that refused everything.
    """

    def _monitor(self, *, has_progress: bool):
        monitor = super()._monitor(has_progress=has_progress)
        self._depart(monitor, has_progress=has_progress)
        return monitor

    def test_a_progress_ramp_that_stalls_completes(self) -> None:
        monitor = self._monitor(has_progress=True)
        for index in range(60):
            self._tick(
                monitor, progress=min(1.0, index / 4.0), offset=self.DEPARTED
            )
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
            self._tick(monitor, progress=0.3, offset=self.DEPARTED)
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
            self._tick(monitor, progress=0.6, offset=self.DEPARTED)
        self.assertFalse(
            self.events["exit_early"],
            "a mid-task pause must not complete the stage: p is 0.6, not >= 0.95",
        )
        self.assertGreater(self.now, 2.0)

        # 2. It resumes and travels, with p now above the threshold. The arm's
        #    peak-to-peak over the window is what refuses here -- |cmd - meas|
        #    is zero on every tick, because the mock arm follows perfectly.
        offset = self.DEPARTED
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
            self._tick(monitor, progress=1.0, offset=self.DEPARTED)
            if self.events["exit_early"]:
                fired_at = self.now
                break
        self.assertIsNotNone(fired_at)
        self.assertGreaterEqual(fired_at, self.params.p10_s)
        self.assertGreaterEqual(fired_at, self.settings.stall_s)

    def test_a_commanded_base_velocity_blocks_completion(self) -> None:
        monitor = self._monitor(has_progress=True)
        for _ in range(60):
            self._tick(monitor, progress=1.0, offset=self.DEPARTED, base=0.2)
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
        self._tick(monitor, progress=1.0, offset=self.DEPARTED, dt=1 / 21.0)
        # The loop blocks for 5 s (a RealSense async_read), then one tick.
        self._tick(monitor, progress=1.0, offset=self.DEPARTED, dt=5.0)
        self.assertFalse(
            self.events["exit_early"],
            "two samples spanning a 5 s gap are not a covered stall window",
        )

    def test_a_sixteen_d_model_completes_by_stall_plus_end_pose(self) -> None:
        monitor = self._monitor(has_progress=False)
        # end_poses[0] of the mock file is start_pose + 0.05, tol 0.12 -- which
        # is WITHIN the tolerance of the start pose itself, the C-1 geometry.
        # Out to 0.3, then back to the end pose and stop. (The departure latch is
        # already set by `_depart`, so what this test measures is the NN rule and
        # the stall window, not the premise; standing at 0.05 from the first tick
        # is the latch's own case and has its own test in DepartureLatchTest.)
        for _ in range(8):
            self._tick(monitor, offset=0.3)
        self.assertFalse(self.events["exit_early"])
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
            self._tick(monitor, progress=1.0, offset=self.DEPARTED)
            if self.events["exit_early"]:
                break
        self.assertTrue(self.events["exit_early"])
        monitor.end_stage()
        self.events["exit_early"] = False

        monitor.begin_stage(self.params, has_progress=True, started=self.now)
        self._tick(monitor, progress=1.0, offset=self.DEPARTED)
        self.assertFalse(
            self.events["exit_early"],
            "the previous stage's stalled window must not complete this stage "
            "on its first tick",
        )

    def test_a_missing_progress_key_makes_the_monitor_blind_not_wrong(self) -> None:
        monitor = self._monitor(has_progress=True)
        for _ in range(60):
            self._tick(
                monitor, progress=None, offset=self.DEPARTED
            )  # 17-D declared, key absent
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

    def test_a_steady_tracking_error_is_still_a_stopped_arm(self) -> None:
        """M-2: 0.06 rad of LAG is not 0.06 rad of travel.

        The arm sits still at a fixed offset while its measurement trails the
        command by a constant 0.06 rad -- what a loaded arm holding against
        gravity does. With one knob for both quantities (0.05) this ran to its
        timeout forever, because every tick looked like it was "still moving".
        """
        monitor = self._monitor(has_progress=True)
        for _ in range(40):
            self._tick(
                monitor, progress=1.0, offset=self.DEPARTED, track=0.06
            )
            if self.events["exit_early"]:
                break
        self.assertTrue(
            self.events["exit_early"],
            "a steady-state following error below stall_track_rad (0.08) must "
            "not block completion; only MEASURED travel above stall_arm_rad "
            "(0.05) may",
        )
        result = monitor.end_stage()
        self.assertEqual(result.reason, comp.REASON_PROGRESS)

    def test_a_tracking_error_over_the_track_knob_still_blocks(self) -> None:
        """The split loosened one quantity, it did not remove the check.

        ``offset = DEPARTED + track`` so the MEASURED arm is still DEPARTED rad
        from the start pose: otherwise 0.2 - 0.12 = 0.08 never latches the
        departure and this would be a test of the latch wearing a stall test's
        name.
        """
        monitor = self._monitor(has_progress=True)
        for _ in range(60):
            self._tick(
                monitor,
                progress=1.0,
                offset=self.DEPARTED + 0.12,
                track=0.12,
            )
        self.assertTrue(
            monitor._stage.departed,
            "the departure must NOT be what is refusing here",
        )
        self.assertFalse(
            self.events["exit_early"],
            "an arm 0.12 rad behind its command is not following it",
        )
        self.assertIsNone(monitor.end_stage())


class DepartureLatchTest(_MonitorHarness, unittest.TestCase):
    """C-1: completion is refused until the stage leaves its start scene.

    Shares the harness, not the premise: here the offset IS the variable, and
    ``_monitor`` is the harness's own (no ``_depart`` prelude).

    EVERY SEQUENCE THAT MEANS TO MOVE STARTS WITH A TICK AT OFFSET 0. The origin
    of the departure measurement is the stage's first MEASURED pose, not the
    file's designated pose, so a sequence whose first tick is already at 0.3
    describes an arm that STARTED at 0.3 and never moved -- which does not depart,
    correctly. The leading zero-offset tick is what makes the designated pose the
    origin, i.e. the ordinary case where a boundary reset has just driven the arm
    there.
    """

    def test_standing_still_at_the_start_pose_never_completes(self) -> None:
        """The 16-D failure that motivated the latch, in its exact geometry.

        The mock file's ``end_poses[0]`` for stage 1 is ``start_pose + 0.05``
        with ``end_pose_tol_rad`` 0.12, so an arm parked ON the start pose is
        0.05 from a demonstrated end pose -- inside tolerance. In the real
        ``configs/chain/stage_params.json`` t03, t04 and t10 are 0.001 rad from
        their own end poses, which is the same thing with no slack at all.
        Stalled plus near-an-end-pose plus past p10 is therefore all true, and
        the stage still did nothing.
        """
        monitor = self._monitor(has_progress=False)
        for _ in range(80):
            self._tick(monitor, offset=0.0)
        self.assertFalse(
            self.events["exit_early"],
            "a stage whose arm never left the designated start pose must NOT "
            "be declared complete, however near an end pose it is standing",
        )
        self.assertIsNone(monitor.end_stage())
        departure = monitor.last_departure
        self.assertFalse(departure.departed)
        self.assertIsNone(departure.at_s)
        self.assertAlmostEqual(departure.arm_rad, 0.0, places=6)

    def test_the_same_refusal_applies_to_a_seventeen_d_progress_head(self) -> None:
        """p = 1.0 out of the start scene is the same failure, other sensor."""
        monitor = self._monitor(has_progress=True)
        for _ in range(80):
            self._tick(monitor, progress=1.0, offset=0.0)
        self.assertFalse(
            self.events["exit_early"],
            "the latch gates the progress path too: a head reading 1.0 before "
            "the arm has moved is not evidence the stage is done",
        )
        self.assertIsNone(monitor.end_stage())

    def test_an_arm_that_leaves_and_comes_back_may_complete(self) -> None:
        """0.15 rad out, back to the end pose, stop -> completion allowed."""
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.0)  # the origin: the designated start pose
        for _ in range(8):
            self._tick(monitor, offset=0.15)
        for _ in range(40):
            self._tick(monitor, offset=0.05)
            if self.events["exit_early"]:
                break
        self.assertTrue(self.events["exit_early"])
        result = monitor.end_stage()
        self.assertEqual(result.reason, comp.REASON_STALL_NN)
        self.assertIsNotNone(result.departed_s)
        self.assertLess(
            result.departed_s,
            result.elapsed_s,
            "the latch must have set BEFORE the completion it is a premise of",
        )
        self.assertTrue(monitor.last_departure.departed)

    def test_an_arm_just_inside_the_threshold_does_not_latch(self) -> None:
        """0.09 < departure_arm_rad 0.10. A reset lands within tol_rad 0.05."""
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.0)  # the origin: the designated start pose
        for _ in range(80):
            self._tick(monitor, offset=0.09)
        self.assertFalse(self.events["exit_early"])
        monitor.end_stage()
        self.assertFalse(monitor.last_departure.departed)

    def test_the_base_alone_can_latch_it(self) -> None:
        """A stage that turns or drives with no arm motion has still departed.

        t01, t03 and t10 are expected to be like that by their names [추정];
        the rule under test does not depend on which stages they are.

        theta.vel 0.5 rad/s for 10 ticks at 21 Hz integrates to about 0.21 rad,
        past departure_base_rot_rad (0.17). The arm never moves.
        """
        monitor = self._monitor(has_progress=False)
        for _ in range(10):
            self._tick(monitor, offset=0.0, theta=0.5)
        self.assertTrue(
            monitor._stage.departed,
            "a stage that only turned the base has still departed",
        )
        # Now the base stops and the arm holds on an end pose: completion may
        # fire, because the premise is satisfied.
        for _ in range(40):
            self._tick(monitor, offset=0.05)
            if self.events["exit_early"]:
                break
        self.assertTrue(self.events["exit_early"])
        result = monitor.end_stage()
        self.assertEqual(result.reason, comp.REASON_STALL_NN)
        self.assertGreater(abs(monitor.last_departure.base_rot_rad), 0.17)

    def test_the_first_tick_integrates_no_base_distance(self) -> None:
        """dt is the tick-to-tick difference, so the first tick's dt is 0.

        Without that, the first tick would be charged a whole 1/fps of whatever
        velocity it happened to command, and a single spurious tick at the
        start could latch a stage that never moved.
        """
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.0, theta=1.0, dt=1 / 21.0)
        self.assertEqual(monitor._stage.base_rot_rad, 0.0)
        self.assertFalse(monitor._stage.departed)

    def test_a_rocking_base_is_not_a_departure(self) -> None:
        """The threshold is on |INTEGRAL|, not on accumulated absolute rotation.

        Four cycles of 6 ticks each way at 0.5 rad/s: the NET rotation never
        leaves (-0.03, +0.12) rad while the total travelled is over 0.5 rad. The
        base is rocking in place, which is where it started, and the arm has not
        moved -- so nothing departed. Summing |theta.vel|*dt instead would latch
        on a stage that went nowhere, which is the whole failure C-1 is about.
        """
        monitor = self._monitor(has_progress=False)
        for _ in range(4):
            for _ in range(6):
                self._tick(monitor, offset=0.0, theta=0.5)
            for _ in range(6):
                self._tick(monitor, offset=0.0, theta=-0.5)
        self.assertFalse(monitor._stage.departed)
        monitor.end_stage()
        self.assertLess(abs(monitor.last_departure.base_rot_rad), 0.17)
        # And it really did keep commanding the base for ~0.5 rad of travel.
        self.assertGreater(self.now, 2.0)

    def test_a_driven_base_can_latch_it_too(self) -> None:
        """x.vel: 0.3 m/s for 10 ticks at 21 Hz is about 0.13 m > 0.10 m."""
        monitor = self._monitor(has_progress=False)
        for _ in range(10):
            self._tick(monitor, offset=0.0, base=0.3)
        self.assertTrue(monitor._stage.departed)
        monitor.end_stage()
        self.assertGreater(abs(monitor.last_departure.base_fwd_m), 0.10)

    def test_the_latch_is_cleared_by_begin_stage(self) -> None:
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.0)  # the origin: the designated start pose
        for _ in range(8):
            self._tick(monitor, offset=0.3)
        self.assertTrue(monitor._stage.departed)
        monitor.end_stage()
        monitor.begin_stage(self.params, has_progress=False, started=self.now)
        self._tick(monitor, offset=0.0)
        self.assertFalse(
            monitor._stage.departed,
            "the previous stage's departure must not carry into this one -- the "
            "robot action pipeline is the one record_loop never resets",
        )

    def test_a_missing_progress_key_does_not_hide_a_real_departure(self) -> None:
        """Blind on the PROGRESS slot; the arm and the base are still readable.

        The departure update runs before the progress-key check for exactly
        this: if it ran after, a blind stage would report "never departed,
        worst arm 0.000 rad" -- a number nothing measured -- and the report
        would send the operator to the start scene while the real fault is that
        the monitor could not read its signal.
        """
        monitor = self._monitor(has_progress=True)
        # The origin, with the progress key PRESENT so this tick is not the one
        # that blinds the monitor; the arm then moves 0.3 rad with it absent.
        self._tick(monitor, progress=0.0, offset=0.0)
        for _ in range(40):
            self._tick(monitor, progress=None, offset=0.3)
        self.assertFalse(
            self.events["exit_early"], "a blind monitor never declares complete"
        )
        self.assertIsNone(monitor.end_stage())
        departure = monitor.last_departure
        self.assertTrue(
            departure.departed,
            "the arm WAS readable and it moved 0.3 rad; only `progress` was "
            "missing",
        )
        self.assertFalse(departure.unknown)
        self.assertAlmostEqual(departure.arm_rad, 0.3, places=6)

    def test_an_unreadable_arm_makes_the_departure_unknown_not_false(self) -> None:
        """A transition the monitor cannot parse at all.

        `departed` is False because nothing latched it, but `unknown` is True,
        and the executor and the report both read THAT: the stage's timeout
        reason must not say `never_departed` and the table must print `?`.
        """
        monitor = self._monitor(has_progress=True)
        for _ in range(5):
            self.now += 1 / 21.0
            monitor({})
        self.assertFalse(self.events["exit_early"])
        self.assertIsNone(monitor.end_stage())
        departure = monitor.last_departure
        self.assertFalse(departure.departed)
        self.assertTrue(departure.unknown)
        self.assertTrue(departure.blind)
        self.assertTrue(departure.as_detail()["departure_unknown"])

    def test_an_unknown_departure_is_not_reported_as_never_departed(self) -> None:
        from stage_runner import executors
        from stage_runner.completion import Departure

        unknown = Departure(
            departed=False,
            at_s=None,
            arm_rad=0.0,
            base_rot_rad=0.0,
            base_fwd_m=0.0,
            blind="the transition carried no action/observation dict",
        )
        _, reason, _ = executors._classify_chain_policy(
            context=_FakeContext(),
            stage=None,
            plan=_FakePlan(control_time_s=10.0),
            elapsed_s=10.0,
            completion=None,
            departure=unknown,
            allow_manual=True,
        )
        self.assertNotIn(comp.REASON_NEVER_DEPARTED, reason)
        self.assertIn("BLIND", reason)
        self.assertIn("UNKNOWN", reason)

    def test_a_measured_non_departure_still_says_never_departed(self) -> None:
        from stage_runner import executors
        from stage_runner.completion import Departure

        measured = Departure(
            departed=False, at_s=None, arm_rad=0.02, base_rot_rad=0.0,
            base_fwd_m=0.0,
        )
        _, reason, _ = executors._classify_chain_policy(
            context=_FakeContext(),
            stage=None,
            plan=_FakePlan(control_time_s=10.0),
            elapsed_s=10.0,
            completion=None,
            departure=measured,
            allow_manual=True,
        )
        self.assertIn(comp.REASON_NEVER_DEPARTED, reason)

    def test_the_departure_survives_window_pruning(self) -> None:
        """_prune drops samples older than the longest window; the latch is not one.

        The arm departs at the start and is then still for ten times stall_s.
        By the time completion is possible every sample that proved it moved has
        been pruned, so a latch derived from `samples` would have forgotten.
        """
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.0)  # the origin: the designated start pose
        for _ in range(4):
            self._tick(monitor, offset=0.3)
        for _ in range(200):
            self._tick(monitor, offset=0.05)
            if self.events["exit_early"]:
                break
        self.assertTrue(self.events["exit_early"])
        self.assertLess(
            len(monitor._stage.samples),
            200,
            "the sample list must be pruned, or this test proves nothing",
        )
        self.assertIsNotNone(monitor.end_stage())


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


class TemporalEnsembleGateTest(NoRobotSdkMixin, unittest.TestCase):
    """M-4: refuse ``temporal_ensemble_coeff`` with ``n_action_steps > 1``.

    The combination is one ACT refuses in its own ``__post_init__``, and the
    chain walks around that refusal by assigning ``n_action_steps`` AFTER
    ``from_pretrained``. The checkpoint below is exactly the bypass: coeff set
    and steps 1, which is VALID on disk and becomes invalid the moment the
    runner widens it.
    """

    def _config(self, directory: Path, *, n_action_steps, coefficient=0.01):
        import tempfile as _tempfile  # noqa: F401  (directory is the caller's)

        from stage_runner import config as cfg_module
        from stage_runner.mock_robot import MockRobotConfig

        checkpoint = directory / "ckpt"
        checkpoint.mkdir(parents=True, exist_ok=True)
        document: dict = {"type": "act", "chunk_size": 30, "n_action_steps": 1}
        if coefficient is not None:
            document["temporal_ensemble_coeff"] = coefficient
        (checkpoint / "config.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        chain = cfg_module.ChainConfig(
            enabled=True,
            params_path=str(MOCK_PARAMS),
            model=cfg_module.ChainModelConfig(
                policy_path=str(checkpoint),
                onehot_k=None,
                n_action_steps=n_action_steps,
            ),
        )
        config = cfg_module.StageRunnerConfig(
            robot=MockRobotConfig(), version=2, chain=chain
        )
        config.stages = cfg_module.expand_chain(config, load_params())
        return config

    def test_a_coefficient_with_thirty_exec_steps_is_refused(self) -> None:
        import tempfile

        from stage_runner import preflight

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), n_action_steps=30)
            with self.assertRaises(preflight.PreflightError) as caught:
                preflight.check_chain_definitions(config, load_params())
        message = str(caught.exception)
        self.assertIn("temporal_ensemble_coeff", message)
        self.assertIn("n_action_steps", message)

    def test_the_gate_reads_the_effective_value_not_the_file(self) -> None:
        """`n_action_steps: 1` in the YAML is the one safe way to run it."""
        import tempfile

        from stage_runner import preflight

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), n_action_steps=1)
            self.assertIsNone(preflight.temporal_ensemble_problem(config))
            # And `null` takes the checkpoint's own 1, which is also safe.
            config.chain.model.n_action_steps = None
            self.assertIsNone(preflight.temporal_ensemble_problem(config))

    def test_no_coefficient_is_no_problem_at_any_exec(self) -> None:
        import tempfile

        from stage_runner import preflight

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(
                Path(directory), n_action_steps=30, coefficient=None
            )
            self.assertIsNone(preflight.temporal_ensemble_problem(config))
            preflight.check_chain_definitions(config, load_params())

    def test_a_loaded_config_object_is_read_the_same_way(self) -> None:
        """cli.main hands over the PreTrainedConfig it already fetched."""
        import tempfile

        from stage_runner import preflight

        class Loaded:
            temporal_ensemble_coeff = 0.01
            n_action_steps = 1

        with tempfile.TemporaryDirectory() as directory:
            # The file says nothing; the object is what is read.
            config = self._config(
                Path(directory), n_action_steps=None, coefficient=None
            )
            problem = preflight.temporal_ensemble_problem(config, Loaded())
            self.assertIsNone(
                problem, "the checkpoint's own 1 step is safe"
            )
            config.chain.model.n_action_steps = 30
            problem = preflight.temporal_ensemble_problem(config, Loaded())
            self.assertIsNotNone(problem)
            self.assertIn("temporal_ensemble_coeff", problem)

    def test_a_hub_id_is_not_guessed_at(self) -> None:
        """No local config.json, no opinion. The gate must not block a hub run."""
        from stage_runner import config as cfg_module
        from stage_runner import preflight
        from stage_runner.mock_robot import MockRobotConfig

        chain = cfg_module.ChainConfig(
            enabled=True,
            params_path=str(MOCK_PARAMS),
            model=cfg_module.ChainModelConfig(
                policy_path="kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000",
                n_action_steps=30,
            ),
        )
        config = cfg_module.StageRunnerConfig(
            robot=MockRobotConfig(), version=2, chain=chain
        )
        self.assertIsNone(preflight.temporal_ensemble_problem(config))


class ResetTerminationTest(NoRobotSdkMixin, unittest.TestCase):
    """M-5: `not_reached` is the OUTCOME of three different events."""

    class _Policy:
        def __init__(self, *, reached=False, refused="", worst=0.42) -> None:
            self.reached = reached
            self.refused = refused
            self.worst_error_rad = worst

    def _plan(self, control_time_s: float = 9.0):
        anchor = {name: 0.0 for name in cp.ARM_JOINT_NAMES}
        return rp.plan_reset(
            stage_number=5,
            anchor=anchor,
            target=dict(anchor, left_joint_1=0.5),
            fps=21,
            settings=rp.ResetSettings(),
        )

    def _classify(self, *, events=None, policy=None, elapsed_s=9.0):
        from stage_runner import executors

        plan = self._plan()
        return executors.classify_chain_reset(
            events=events or {},
            policy=policy or self._Policy(),
            plan=plan,
            elapsed_s=elapsed_s,
            margin_s=0.5,
            tol_rad=0.05,
        )

    def test_a_right_arrow_during_the_ramp_is_manual_interrupt(self) -> None:
        from stage_runner import executors

        plan = self._plan()
        terminated_by, reason, end_reason = self._classify(
            elapsed_s=1.0, events={"exit_early": True}
        )
        self.assertEqual(
            terminated_by,
            TERMINATED_BY_NOT_REACHED,
            "the OUTCOME is unchanged -- the arms are genuinely not at the next "
            "stage's start pose, so the chain still fails",
        )
        self.assertEqual(
            end_reason, executors.RESET_REASON_MANUAL_INTERRUPT
        )
        self.assertIn("CUT SHORT", reason)
        self.assertLess(1.0, plan.control_time_s - 0.5)

    def test_running_the_ceiling_out_is_not_manual_interrupt(self) -> None:
        from stage_runner import executors

        plan = self._plan()
        terminated_by, reason, end_reason = self._classify(
            elapsed_s=plan.control_time_s
        )
        self.assertEqual(terminated_by, TERMINATED_BY_NOT_REACHED)
        self.assertEqual(end_reason, executors.RESET_REASON_CEILING)
        self.assertIn("NO RETRY", reason)

    def test_arrival_outranks_the_elapsed_heuristic(self) -> None:
        """A reset that arrives EARLY also ends early. It is not an interrupt."""
        from stage_runner import executors

        terminated_by, _, end_reason = self._classify(
            elapsed_s=1.0,
            policy=self._Policy(reached=True, worst=0.01),
            events={"exit_early": True},
        )
        self.assertEqual(terminated_by, TERMINATED_BY_REACHED)
        self.assertEqual(end_reason, executors.RESET_REASON_REACHED)

    def test_a_refusal_outranks_arrival_and_the_heuristic(self) -> None:
        from stage_runner import executors

        terminated_by, reason, end_reason = self._classify(
            elapsed_s=0.2, policy=self._Policy(refused="no counterpart")
        )
        self.assertEqual(terminated_by, TERMINATED_BY_NOT_REACHED)
        self.assertEqual(end_reason, executors.RESET_REASON_REFUSED)
        self.assertIn("refused mid-entry", reason)

    def test_an_escape_outranks_everything(self) -> None:
        from stage_runner.results import TERMINATED_BY_STOP_RECORDING

        terminated_by, _, _ = self._classify(
            elapsed_s=0.2,
            events={"stop_recording": True, "exit_early": True},
            policy=self._Policy(reached=True),
        )
        self.assertEqual(terminated_by, TERMINATED_BY_STOP_RECORDING)


class ResetExecutorRefusalTest(NoRobotSdkMixin, unittest.TestCase):
    """C-2 through the executor: refused BEFORE anything was commanded."""

    def _context(self, robot):
        import types

        from stage_runner import config as cfg_module
        from stage_runner.completion import CompletionMonitorStep
        from stage_runner.context import ChainRuntime

        params = load_params()
        action_names = [f"{name}.pos" for name in REALISTIC_JOINT_NAMES] + [
            "x.vel",
            "theta.vel",
        ]
        state_names = [f"{name}.pos" for name in REALISTIC_JOINT_NAMES]
        reset_config = cfg_module.ResetConfig()
        completion_config = cfg_module.CompletionConfig()
        events: dict[str, bool] = {}
        chain = ChainRuntime(
            params=params,
            monitor=CompletionMonitorStep(events),
            completion=completion_config.to_settings(),
            reset=reset_config.to_settings(),
            has_progress=False,
            action_names=tuple(action_names),
            state_names=tuple(state_names),
            onehot=None,
            onehot_k=None,
            allow_manual_complete=True,
        )
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                dataset=types.SimpleNamespace(fps=21), display_data=False
            ),
            robot=robot,
            events=events,
            chain=chain,
            buffered_frame_count=lambda: 0,
            # Only reached on the path where call_record_loop is stubbed out.
            processors=None,
            dataset=None,
        )

    def _robot(self, **positions):
        from stage_runner.mock_robot import MockRobot, MockRobotConfig

        robot = MockRobot(
            MockRobotConfig(
                joint_count=14,
                realistic_joint_names=True,
                include_base_in_state=False,
            )
        )
        robot.connect()
        robot._joint_positions.update(positions)
        return robot

    def _stage(self, *, initial: bool):
        from stage_runner import config as cfg_module

        return cfg_module.StageConfig(
            id=cfg_module.reset_stage_id(1, initial=initial),
            name="reset to task01",
            executor=cfg_module.EXECUTOR_CHAIN_RESET,
            instruction="reset:01",
            kind=cfg_module.STAGE_KIND_RESET,
            stage_number=1,
            initial_reset=initial,
        )

    def test_an_initial_reset_from_far_away_commands_nothing(self) -> None:
        from stage_runner import executors

        # Stage 1's designated left_joint_0 is 0.0 in the mock file; put the arm
        # 0.8 rad away, which is inside max_jump_rad (1.5) and outside
        # initial_max_jump_rad (0.6).
        robot = self._robot(left_joint_0=0.8)
        context = self._context(robot)
        result = executors.run_chain_reset_stage(context, self._stage(initial=True))
        robot.disconnect()

        self.assertEqual(result.terminated_by, TERMINATED_BY_NOT_REACHED)
        self.assertEqual(
            result.frames,
            0,
            "refused before record_loop was entered, so no frame was recorded",
        )
        self.assertEqual(result.elapsed_s, 0.0)
        self.assertEqual(
            robot.sent_actions,
            [],
            "NOTHING was commanded -- this is the assertion that matters, "
            "because a refusal that had already sent one tick would have moved "
            "the arms toward a pose the runner then refused to drive to",
        )
        self.assertIn("initial_max_jump_rad", result.reason)
        self.assertIn("0.3 rad", result.reason)

    def test_the_same_pose_at_a_boundary_is_planned_and_entered(self) -> None:
        """Contrast: 0.8 rad is an ordinary boundary gap, not a refusal.

        ``call_record_loop`` is stubbed out, so the ramp is planned and the
        bundle built but no tick runs -- which lands in the manual_interrupt
        branch (elapsed ~0 with no arrival). That is the point: the wiring
        reaches the classifier, and the classifier names what happened.
        """
        from stage_runner import executors, record_adapter

        robot = self._robot(left_joint_0=0.8)
        context = self._context(robot)
        original = record_adapter.call_record_loop
        record_adapter.call_record_loop = lambda **kwargs: None
        try:
            result = executors.run_chain_reset_stage(
                context, self._stage(initial=False)
            )
        finally:
            record_adapter.call_record_loop = original
        robot.disconnect()

        self.assertEqual(result.terminated_by, TERMINATED_BY_NOT_REACHED)
        self.assertEqual(
            result.detail["reset_end_reason"],
            executors.RESET_REASON_MANUAL_INTERRUPT,
        )
        self.assertAlmostEqual(result.detail["reset_dmax"], 0.8, places=6)
        self.assertEqual(result.detail["reset_jump_limit_rad"], 1.5)
        self.assertFalse(result.detail["reset_initial"])


class BaseFallbackTest(NoRobotSdkMixin, unittest.TestCase):
    """The direct base command, which only the uncollected pytest files covered.

    ``test_smoke_runner.py`` exercises this path, but it is pytest-style bare
    functions and ``unittest discover`` collects ZERO tests from it -- so in the
    suite that actually runs, the fallback's SUCCESS path was never executed.
    """

    class _Base:
        """The smallest thing that behaves like MobileAIRobot.base.

        ``TrossenSlate.set_cmd_vel`` returns a bool and mobileai.py:554/:569
        both test it that way, which makes it a real delivery signal -- unlike
        ``send_action``'s echo, which carries the commanded values either way.
        """

        def __init__(self, robot, *, accepts=True) -> None:
            self.robot = robot
            self.accepts = accepts
            self.commands: list[tuple[float, float]] = []

        def set_cmd_vel(self, x_vel: float, theta_vel: float) -> bool:
            self.commands.append((float(x_vel), float(theta_vel)))
            if not self.accepts:
                return False
            self.robot._base_velocity = {
                "x.vel": float(x_vel),
                "theta.vel": float(theta_vel),
            }
            return True

    def _robot(self, *, with_base=True, accepts=True, driving=True):
        from stage_runner.mock_robot import MockRobot, MockRobotConfig

        robot = MockRobot(
            MockRobotConfig(joint_count=14, realistic_joint_names=True)
        )
        robot.connect()
        if driving:
            robot._base_velocity = {"x.vel": 0.4, "theta.vel": 0.0}
        if with_base:
            robot.base = self._Base(robot, accepts=accepts)
        return robot

    def _break_observation(self, robot):
        def get_observation():
            # The measured fault this path exists for: MobileAIRobot reads all
            # three RealSense cameras through an unguarded async_read
            # (mobileai.py:526-528), so a dead camera takes out the hold action
            # the primary stop needs.
            raise RuntimeError("RealSense async_read timed out (injected)")

        robot.get_observation = get_observation

    def test_zero_base_directly_succeeds_and_reports_empty(self) -> None:
        from stage_runner import transitions

        robot = self._robot()
        self.assertEqual(transitions.zero_base_directly(robot), "")
        self.assertEqual(robot.base.commands, [(0.0, 0.0)])
        self.assertEqual(robot._base_velocity, {"x.vel": 0.0, "theta.vel": 0.0})
        robot.disconnect()

    def test_a_dead_observation_falls_back_and_the_base_stops(self) -> None:
        from stage_runner import transitions

        robot = self._robot()
        self._break_observation(robot)
        outcome = transitions.stop_base(robot)
        self.assertEqual(outcome.path, transitions.STOP_PATH_FALLBACK)
        self.assertTrue(outcome.base_is_stopped)
        self.assertEqual(outcome.reason, "stop_base_fallback")
        self.assertIsNone(
            outcome.action,
            "the fallback builds no action dict -- not commanding the arms is "
            "the point of it, and they hold their position anyway",
        )
        self.assertEqual(robot.base.commands, [(0.0, 0.0)])
        self.assertEqual(robot._base_velocity, {"x.vel": 0.0, "theta.vel": 0.0})
        self.assertEqual(
            robot.sent_actions,
            [],
            "send_action was never reached, so the arms were not commanded",
        )

    def test_a_base_that_refuses_the_transaction_is_a_failed_stop(self) -> None:
        from stage_runner import transitions

        robot = self._robot(accepts=False)
        self._break_observation(robot)
        outcome = transitions.stop_base(robot)
        self.assertEqual(outcome.path, transitions.STOP_PATH_FAILED)
        self.assertFalse(
            outcome.base_is_stopped, "cli.main turns this into exit code 3"
        )
        self.assertIn("returned False", outcome.error)

    def test_a_robot_with_no_base_has_no_fallback(self) -> None:
        """MockRobot ships WITHOUT `.base`; the option is opt-in per test."""
        from stage_runner import transitions

        robot = self._robot(with_base=False)
        self.assertFalse(hasattr(robot, "base"))
        self.assertIn("no .base", transitions.zero_base_directly(robot))

    def test_a_keyboard_interrupt_reaches_the_operator_after_the_stop(self) -> None:
        from stage_runner import transitions

        robot = self._robot()

        def get_observation():
            raise KeyboardInterrupt

        robot.get_observation = get_observation
        with self.assertRaises(KeyboardInterrupt):
            transitions.stop_base(robot)
        self.assertEqual(
            robot.base.commands,
            [(0.0, 0.0)],
            "the fallback must run BEFORE the interrupt is re-raised",
        )
        # And the caller that is already unwinding gets the truth instead.
        robot.base.commands.clear()
        outcome = transitions.stop_base(robot, reraise_interrupt=False)
        self.assertEqual(outcome.path, transitions.STOP_PATH_FALLBACK)


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

    The module itself imports only ``torch``, ``lerobot.processor.pipeline`` and
    ``lerobot.configs.types`` (enums and one dataclass, for the round-2 env
    feature), so no stub is needed for it; the stub below exists so the PARENT
    package name resolves without being executed.
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


# --------------------------------------------------------------------------- #
# round 2: the same one-hot ALSO as its own ACT encoder token
# (observation.environment_state). Still loaded by PATH -- no robot SDK.
# --------------------------------------------------------------------------- #

_ENV_KEY = "observation.environment_state"
_STATE_KEY = "observation.state"


def _policy_cfg(state_dim: int | None = 27, env_dim: int | None = None, env_type="ENV"):
    """A duck-typed stand-in for the checkpoint's ``PreTrainedConfig``.

    Only ``input_features`` is read by the patch, and only ``shape``/``type`` off
    each feature -- a real ``ACTConfig`` would need the whole checkpoint. Built
    as a function rather than a class per case so the ENV-vs-no-ENV pair differs
    in exactly one argument, which is the behaviour under test.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature

    features: dict[str, object] = {}
    if state_dim is not None:
        features[_STATE_KEY] = PolicyFeature(
            type=FeatureType.STATE, shape=(state_dim,)
        )
    if env_dim is not None:
        features[_ENV_KEY] = PolicyFeature(type=env_type, shape=(env_dim,))

    class Config:
        input_features = features

    return Config()


class EnvTokenStepTest(unittest.TestCase):
    """The round-2 env token: same vector, second key, nothing else moves.

    No NoRobotSdkMixin, for the same reason ``TaskOneHotStepTest`` has none: the
    module under test is loaded by path with the package stubbed under another
    name, and the real package's absence is asserted inline.
    """

    def _step(self, **kwargs):
        module = _load_task_onehot_module()
        self.assertNotIn("trossen_slate", sys.modules)
        self.assertNotIn("lerobot_robot_trossen", sys.modules)
        return module, module.insert_task_onehot(
            self._preprocessor(module), _policy_cfg(**kwargs), 1, 11
        )

    @staticmethod
    def _preprocessor(module):
        from lerobot.processor.normalize_processor import NormalizerProcessorStep

        class Preprocessor:
            def __init__(self) -> None:
                self.steps = [object.__new__(NormalizerProcessorStep)]

        return Preprocessor()

    def _run(self, step, state):
        from lerobot.processor import create_transition
        from lerobot.processor.core import TransitionKey

        out = step(create_transition(observation={_STATE_KEY: state.clone()}))
        return out[TransitionKey.OBSERVATION]

    def test_env_feature_makes_both_keys_the_same_one_hot(self) -> None:
        import torch

        _, step = self._step(state_dim=27, env_dim=11)
        self.assertEqual(step.env_k, 11)

        obs = self._run(step, torch.zeros((1, 14), dtype=torch.float32))
        self.assertIn(_ENV_KEY, obs)
        state, env = obs[_STATE_KEY], obs[_ENV_KEY]
        self.assertEqual(tuple(state.shape), (1, 27))
        self.assertEqual(
            tuple(env.shape),
            (1, 11),
            "ACT projects (B, K) through encoder_env_state_input_proj; the token "
            "must carry the batch dimension the state carries",
        )
        self.assertTrue(
            torch.equal(env, state[..., 16:]),
            "the env token IS the state's one-hot tail -- not a second signal",
        )
        self.assertEqual(env.dtype, state.dtype)
        self.assertEqual(env.device, state.device)
        self.assertFalse(
            env.data_ptr() == state.data_ptr(),
            "cloned, not aliased: an in-place step must not move one copy only",
        )

    def test_a_round_1_checkpoint_gets_no_env_key_and_the_same_state(self) -> None:
        import torch

        _, with_env = self._step(state_dim=27, env_dim=11)
        _, without = self._step(state_dim=27)
        self.assertIsNone(without.env_k)

        raw = torch.arange(14, dtype=torch.float32).reshape(1, 14)
        plain = self._run(without, raw)
        self.assertNotIn(
            _ENV_KEY,
            plain,
            "a round-1 checkpoint (M1/M2/M3, tph) declares no ENV feature, and "
            "adding the key anyway would hand ACT a tensor it never looks up "
            "while changing a path that is in production",
        )
        self.assertTrue(
            torch.equal(plain[_STATE_KEY], self._run(with_env, raw)[_STATE_KEY]),
            "observation.state must be identical in both rounds",
        )

    def test_set_stage_moves_both_keys(self) -> None:
        import torch

        _, step = self._step(state_dim=27, env_dim=11)
        state = torch.zeros((1, 14), dtype=torch.float32)

        obs = self._run(step, state)
        self.assertEqual(int(torch.argmax(obs[_ENV_KEY][0]).item()), 0)

        step.set_stage(7)
        obs = self._run(step, state)
        self.assertEqual(int(torch.argmax(obs[_STATE_KEY][0, 16:]).item()), 6)
        self.assertEqual(
            int(torch.argmax(obs[_ENV_KEY][0]).item()),
            6,
            "both places the one-hot appears are built from one `hot` tensor, so "
            "set_stage cannot move one and leave the other",
        )
        self.assertTrue(torch.equal(obs[_ENV_KEY], obs[_STATE_KEY][..., 16:]))

    def test_the_announcement_carries_the_env_marker(self) -> None:
        """``(+env token)`` is what the operator is told to look for.

        docs/eval_najy.md G says its ABSENCE means a round-1 checkpoint is
        running, so the string itself is an interface.
        """
        import logging

        import torch

        module, step = self._step(state_dim=27, env_dim=11)
        with self.assertLogs(module.logger, level=logging.INFO) as caught:
            self._run(step, torch.zeros((1, 14), dtype=torch.float32))
        line = "\n".join(caught.output)
        self.assertIn("stage 1/11 active", line)
        self.assertIn("(+env token)", line)

        _, round1 = self._step(state_dim=27)
        with self.assertLogs(module.logger, level=logging.INFO) as caught:
            self._run(round1, torch.zeros((1, 14), dtype=torch.float32))
        self.assertNotIn("(+env token)", "\n".join(caught.output))

    def test_a_mismatched_env_width_is_refused_before_the_robot_connects(self) -> None:
        module = _load_task_onehot_module()
        for env_dim in (10, 12):
            with self.assertRaises(RuntimeError):
                module.insert_task_onehot(
                    self._preprocessor(module),
                    _policy_cfg(state_dim=27, env_dim=env_dim),
                    1,
                    11,
                )
        # Constructing the step directly is guarded too -- insert_task_onehot is
        # not the only caller the class has to survive.
        with self.assertRaises(RuntimeError):
            module.TaskOneHotStep(stage=1, k=11, expected_dim=27, env_k=10)

    def test_a_non_env_type_is_refused(self) -> None:
        module = _load_task_onehot_module()
        with self.assertRaises(RuntimeError):
            # Declared as STATE, the normalizer would MEAN_STD it and the token
            # would silently stop being a one-hot.
            module.insert_task_onehot(
                self._preprocessor(module),
                _policy_cfg(state_dim=27, env_dim=11, env_type="STATE"),
                1,
                11,
            )

    def test_transform_features_declares_the_env_feature(self) -> None:
        from lerobot.configs.types import (
            FeatureType,
            PipelineFeatureType,
            PolicyFeature,
        )

        module = _load_task_onehot_module()
        initial = {
            PipelineFeatureType.OBSERVATION: {
                _STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(14,))
            },
            PipelineFeatureType.ACTION: {},
        }

        _, step = self._step(state_dim=27, env_dim=11)
        out = step.transform_features(initial)
        declared = out[PipelineFeatureType.OBSERVATION][_ENV_KEY]
        self.assertEqual(declared.type, FeatureType.ENV)
        self.assertEqual(tuple(declared.shape), (11,))
        self.assertNotIn(
            _ENV_KEY,
            initial[PipelineFeatureType.OBSERVATION],
            "the pipeline hands the same dict to every step (pipeline.py:1332), "
            "so this must not mutate its input",
        )

        _, round1 = self._step(state_dim=27)
        self.assertIs(round1.transform_features(initial), initial)

    def test_transform_features_never_reaches_the_dataset_features(self) -> None:
        """The recorded schema is the ROBOT's, not the policy preprocessor's.

        ``aggregate_pipeline_dataset_features`` is the only caller of
        ``transform_features`` in 0.4.4 (``pipeline_features.py:91``), and both
        upstream ``record()`` (``lerobot_record.py:449-462``) and this repo's
        chain (``record_adapter.py:229-238``) call it on ``teleop_action`` /
        ``robot_observation`` -- never on the pipeline the one-hot step is
        inserted into. Asserted here because the step DOES declare a feature
        now, so the reason it is harmless has to be pinned down.
        """
        from stage_runner import record_adapter

        source = record_adapter.build_dataset_features.__code__.co_names
        self.assertIn("aggregate_pipeline_dataset_features", source)
        self.assertIn("teleop_action", source)
        self.assertIn("robot_observation", source)
        self.assertNotIn("preprocessor", source)

    def test_the_env_token_survives_a_real_preprocessor_pipeline(self) -> None:
        """End to end through the steps make_act_pre_post_processors builds.

        Calling the step alone proves nothing about the two converters the real
        pipeline runs it between: ``batch_to_transition`` keeps only keys
        prefixed ``observation`` and ``transition_to_batch`` flattens them back
        (``converters.py:354, 396-398``). A key added mid-pipeline has to
        survive both, be left alone by the normalizer (ENV is absent from ACT's
        ``normalization_mapping``, ``configuration_act.py:89-95``), and come out
        in the dict ``select_action`` is handed.

        Step order is ACT's own (``processor_act.py:56-66``) with the one-hot
        step where ``_insert`` puts it: AFTER AddBatchDimension and Device, so
        the state it widens already has its batch dimension and is on device.
        """
        import torch
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.processor import (
            AddBatchDimensionProcessorStep,
            DeviceProcessorStep,
            NormalizerProcessorStep,
            PolicyProcessorPipeline,
        )

        _, step = self._step(state_dim=27, env_dim=11)
        features = {
            _STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(27,)),
            _ENV_KEY: PolicyFeature(type=FeatureType.ENV, shape=(11,)),
        }
        normalizer = NormalizerProcessorStep(
            features=features,
            # ACT's mapping verbatim: no ENV entry, which is what makes the
            # token pass through untouched.
            norm_map={
                FeatureType.VISUAL: NormalizationMode.MEAN_STD,
                FeatureType.STATE: NormalizationMode.MEAN_STD,
                FeatureType.ACTION: NormalizationMode.MEAN_STD,
            },
            stats={
                _STATE_KEY: {
                    "mean": torch.zeros(27),
                    # Not 1.0, so a state that went through the normalizer is
                    # distinguishable from one that did not.
                    "std": torch.full((27,), 2.0),
                }
            },
        )
        pipeline = PolicyProcessorPipeline(
            steps=[
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device="cpu"),
                step,
                normalizer,
            ],
            name="task_onehot_pipeline_under_test",
        )

        step.set_stage(4)
        batch = pipeline({_STATE_KEY: torch.zeros(14, dtype=torch.float32)})

        self.assertIn(_ENV_KEY, batch)
        env, state = batch[_ENV_KEY], batch[_STATE_KEY]
        self.assertEqual(tuple(env.shape), (1, 11))
        self.assertEqual(tuple(state.shape), (1, 27))
        self.assertAlmostEqual(
            float(env[0, 3]),
            1.0,
            msg="ENV has no entry in ACT's normalization_mapping, so the "
            "normalizer leaves the token a one-hot",
        )
        self.assertAlmostEqual(
            float(state[0, 19]),
            0.5,
            msg="the state tail IS normalized (std=2.0), which is exactly why "
            "the env token has to be a separate key rather than a slice",
        )
        self.assertEqual(int(torch.argmax(env[0]).item()), 3)


class EnvTokenPreflightTest(NoRobotSdkMixin, unittest.TestCase):
    """``check_stage_dimensions`` accepts round 2 and refuses the near-misses.

    The gate runs before ``robot.connect()`` and before any weights download, so
    every refusal here is one the operator sees instead of a KeyError inside
    ``record_loop`` on frame 1.
    """

    @staticmethod
    def _robot(include_base: bool = False):
        class Robot:
            name = "mobileai_robot"
            action_features = {
                f"a{i}": float for i in range(16)
            }
            observation_features = {
                **{f"j{i}": float for i in range(14)},
                **({"x.vel": float, "theta.vel": float} if include_base else {}),
                # One camera entry, which robot_state_dimension must exclude.
                "cam": (480, 640, 3),
            }

        return Robot()

    @staticmethod
    def _stage(policy_path: str = "hub/ckpt"):
        class Stage:
            id = "s1"

        Stage.policy_path = policy_path
        return Stage()

    def _check(self, *, state, env, onehot_k, action=16):
        from stage_runner import preflight

        config = _policy_cfg(state_dim=state, env_dim=env)

        from lerobot.configs.types import FeatureType, PolicyFeature

        config.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action,))
        }
        preflight.check_stage_dimensions(
            self._robot(),
            [self._stage()],
            {"s1": config},
            onehot_k=onehot_k,
            allow_extra_action=True,
        )

    def test_env_11_with_state_27_and_k_11_passes(self) -> None:
        self._check(state=27, env=11, onehot_k=11)

    def test_a_round_1_checkpoint_still_passes(self) -> None:
        self._check(state=27, env=None, onehot_k=11)
        self._check(state=14, env=None, onehot_k=None)

    def test_env_11_with_a_state_that_has_no_one_hot_tail_is_refused(self) -> None:
        from stage_runner import preflight

        with self.assertRaises(preflight.PreflightError) as caught:
            self._check(state=16, env=11, onehot_k=11)
        self.assertIn("environment_state", str(caught.exception))

    def test_env_declared_with_onehot_k_null_is_refused(self) -> None:
        from stage_runner import preflight

        with self.assertRaises(preflight.PreflightError) as caught:
            # The dangerous shape: 16-D state matches the robot, so every width
            # check passes and only the unfillable ENV key is wrong -- which ACT
            # discovers on frame 1, after the arms are powered.
            self._check(state=16, env=11, onehot_k=None)
        self.assertIn("onehot_k", str(caught.exception))

    def test_an_env_width_that_is_not_k_is_refused(self) -> None:
        from stage_runner import preflight

        with self.assertRaises(preflight.PreflightError):
            self._check(state=27, env=10, onehot_k=11)

    def test_the_env_reader_is_ordered_before_make_policy(self) -> None:
        """Same contract as ``checkpoint_state_dimension``: read off config.json.

        ``make_policy`` keeps ``input_features`` only because the checkpoint
        declared it (``factory.py:471`` guards the assignment with
        ``if not cfg.input_features``), so this reader works on both sides of
        that call -- but it is called from the gate, which is before it.
        """
        from stage_runner import preflight

        self.assertIsNone(
            preflight.checkpoint_env_state_dimension(_policy_cfg(state_dim=27))
        )
        self.assertEqual(
            preflight.checkpoint_env_state_dimension(
                _policy_cfg(state_dim=27, env_dim=11)
            ),
            11,
        )


# --------------------------------------------------------------------------- #
# 2026-10-06 codex cross-review. One class per finding, numbered in its docstring.
# --------------------------------------------------------------------------- #


class SignalLatchTest(NoRobotSdkMixin, unittest.TestCase):
    """[치명]2: SIGTERM/SIGHUP must reach the teardown, not kill the process.

    NO SIGNAL IS EVER SENT HERE. ``os.kill(os.getpid(), SIGTERM)`` would deliver
    into whatever the unittest runner happens to be doing and the SystemExit would
    unwind out of the test runner itself; the handler is a plain function, so
    calling it is the whole behaviour. What a real delivery adds is only the
    operating system's choice of WHEN, and that is not what these assertions are
    about.
    """

    def setUp(self) -> None:
        from stage_runner import cli

        self.cli = cli
        self.latch = cli.SignalLatch()

    def tearDown(self) -> None:
        self.latch.restore()
        super().tearDown()

    def test_it_traps_exactly_sigterm_and_sighup(self) -> None:
        import signal

        self.assertEqual(
            set(self.cli._TRAPPED_SIGNALS),
            {signal.SIGTERM, signal.SIGHUP},
            "SIGINT is deliberately absent: Python already maps it to "
            "KeyboardInterrupt, which unwinds, and transitions.stop_base has a "
            "dedicated path for it",
        )

    def test_install_replaces_the_handlers_and_restore_puts_them_back(self) -> None:
        import signal

        before = {
            number: signal.getsignal(number) for number in self.cli._TRAPPED_SIGNALS
        }
        self.assertTrue(self.latch.install())
        for number in self.cli._TRAPPED_SIGNALS:
            self.assertNotEqual(signal.getsignal(number), before[number])
        self.latch.restore()
        for number in self.cli._TRAPPED_SIGNALS:
            self.assertEqual(
                signal.getsignal(number),
                before[number],
                "cli.main is called repeatedly in-process (every e2e test does "
                "it), so a handler left pointing at a dead latch would raise "
                "SystemExit for a signal the NEXT run has not seen",
            )

    def test_an_inherited_sig_ign_is_left_alone(self) -> None:
        """WHAT MAKES `nohup` WORK, and the docs promise it.

        ``nohup`` starts the child with SIGHUP set to SIG_IGN; an ignored
        disposition survives ``exec``, and the point is that the run SURVIVES the
        terminal going away. Taking it over with a handler would make a dropped SSH
        session under nohup tear the run down -- the opposite of what nohup was
        asked for.
        """
        import signal

        previous = signal.getsignal(signal.SIGHUP)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            self.assertTrue(self.latch.install())
            self.assertIs(
                signal.getsignal(signal.SIGHUP),
                signal.SIG_IGN,
                "an inherited SIG_IGN must not be replaced",
            )
            self.assertNotEqual(
                signal.getsignal(signal.SIGTERM),
                signal.SIG_IGN,
                "and SIGTERM, which nohup does NOT ignore, is still trapped",
            )
            self.latch.restore()
            self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
        finally:
            signal.signal(signal.SIGHUP, previous)

    def test_a_signal_inside_the_teardown_never_raises(self) -> None:
        """The failure the latch could cause through the latch.

        cli's ``finally`` wraps ``robot.disconnect()`` in ``except Exception``, and
        a SystemExit is not an Exception -- so a first delivery landing between the
        ``try`` and disconnect()'s own ``base.set_cmd_vel(0.0, 0.0)`` would escape
        and skip the call that stops the base.
        """
        import signal

        events: dict[str, bool] = {}
        self.latch.bind_events(events)
        self.latch.enter_teardown()
        self.assertIsNone(
            self.latch.handle(signal.SIGTERM, None),
            "no raise once the teardown has begun, even for the FIRST signal",
        )
        self.assertTrue(events["stop_recording"], "the flags are still set")
        self.assertEqual(self.latch.signum, int(signal.SIGTERM))

    def test_the_teardown_guard_is_wired_into_cli(self) -> None:
        """A guard the teardown does not call is a comment."""
        self.assertIn(
            "enter_teardown",
            self.cli.run_trial_process.__code__.co_names,
            "run_trial_process's finally must arm it before it touches hardware",
        )

    def test_the_first_delivery_sets_both_flags_and_becomes_system_exit(self) -> None:
        import signal

        events: dict[str, bool] = {}
        self.latch.bind_events(events)
        with self.assertRaises(SystemExit) as caught:
            self.latch.handle(signal.SIGTERM, None)
        self.assertEqual(
            caught.exception.code,
            128 + int(signal.SIGTERM),
            "the conventional shell encoding, so a wrapper script can tell which "
            "signal ended the run",
        )
        self.assertTrue(
            events["stop_recording"],
            "stop_recording and not just exit_early: exit_early alone means "
            "'this stage is done, go on', which would walk the robot into the "
            "next stage",
        )
        self.assertTrue(
            events["exit_early"],
            "record_loop breaks on THIS flag at the top of its next iteration, "
            "which is the graceful half -- the tick in flight finishes and the "
            "ordinary boundary runs",
        )
        self.assertTrue(self.latch.tripped)

    def test_a_second_delivery_does_not_unwind_out_of_the_teardown(self) -> None:
        import signal

        events: dict[str, bool] = {}
        self.latch.bind_events(events)
        with self.assertRaises(SystemExit):
            self.latch.handle(signal.SIGTERM, None)
        events.clear()
        # No raise: by now the finally is zeroing the base and disconnecting, and
        # unwinding out of THAT would leave the arms torqued -- the opposite of
        # what the operator asked for.
        self.assertIsNone(self.latch.handle(signal.SIGHUP, None))
        self.assertEqual(
            self.latch.signum,
            int(signal.SIGTERM),
            "the FIRST signal is the one recorded; the second only re-sets flags",
        )
        self.assertTrue(events["stop_recording"], "the flags are re-set anyway")

    def test_a_signal_before_the_events_dict_exists_still_stops_the_loop(self) -> None:
        """The dict is created at step 11d; the latch is armed long before that."""
        import signal

        with self.assertRaises(SystemExit):
            self.latch.handle(signal.SIGHUP, None)
        events: dict[str, bool] = {}
        self.latch.bind_events(events)
        self.assertTrue(events["stop_recording"])
        self.assertTrue(events["exit_early"])

    def test_main_installs_and_restores_around_the_run(self) -> None:
        """``main`` is a wrapper whose only job is the latch's lifetime."""
        import signal

        before = signal.getsignal(signal.SIGTERM)
        seen: dict[str, object] = {}

        def fake_run(argv, latch):
            seen["latch"] = latch
            seen["handler"] = signal.getsignal(signal.SIGTERM)
            return self.cli.EXIT_OK

        original = self.cli.run_trial_process
        self.cli.run_trial_process = fake_run
        try:
            self.assertEqual(self.cli.main([]), self.cli.EXIT_OK)
        finally:
            self.cli.run_trial_process = original
        self.assertIsInstance(seen["latch"], self.cli.SignalLatch)
        self.assertNotEqual(
            seen["handler"], before, "the latch must be armed DURING the run"
        )
        self.assertEqual(
            signal.getsignal(signal.SIGTERM),
            before,
            "and disarmed after it, on every path out",
        )


class StopBaseConfirmationTest(NoRobotSdkMixin, unittest.TestCase):
    """[중요]3: the primary stop now CONFIRMS through the base's own bool.

    ``MobileAIRobot.send_action`` does not raise when the base write fails -- a
    failed Modbus transaction is a throttled warning (mobileai.py:547-558) and the
    echo carries the sanitized COMMANDED velocities either way -- so every
    boundary where the base silently refused the stop used to be filed as
    ``primary``, i.e. as clean, and the run exited 0.
    """

    _Base = BaseFallbackTest._Base
    _robot = BaseFallbackTest._robot

    def test_an_accepted_confirmation_is_a_clean_primary_stop(self) -> None:
        from stage_runner import transitions

        robot = self._robot()
        outcome = transitions.stop_base(robot)
        self.assertEqual(outcome.path, transitions.STOP_PATH_PRIMARY)
        self.assertIs(
            outcome.direct_ok,
            True,
            "the bool from base.set_cmd_vel is the ONLY delivery ack anywhere in "
            "this module",
        )
        self.assertTrue(outcome.base_is_stopped)
        self.assertIsNotNone(outcome.action, "the hold action went out as well")
        self.assertEqual(
            robot.base.commands,
            [(0.0, 0.0)],
            "exactly one confirming command, after the hold action",
        )
        robot.disconnect()

    def test_a_refused_confirmation_is_not_a_clean_stop(self) -> None:
        from stage_runner import transitions

        robot = self._robot(accepts=False)
        outcome = transitions.stop_base(robot)
        self.assertEqual(outcome.path, transitions.STOP_PATH_PRIMARY_DIRECT_FAILED)
        self.assertIs(outcome.direct_ok, False)
        self.assertFalse(
            outcome.base_is_stopped,
            "cli.main turns this into exit code 3 -- 'go look at the robot'. "
            "Before this path existed the same run exited 0",
        )
        self.assertIsNotNone(
            outcome.action,
            "the hold action DID go out; what failed is the confirmation, and "
            "the two are different facts",
        )
        self.assertEqual(outcome.reason, "stop_base_primary_direct_failed")
        self.assertIn("returned False", outcome.error)
        robot.disconnect()

    def test_a_robot_with_no_base_still_reports_a_clean_primary_stop(self) -> None:
        """NOT APPLICABLE IS NOT A FAILURE, and this is the test that says so.

        MockRobot ships without ``.base``. Treating "there is no base command to
        confirm with" as a failed confirmation would make every boundary of every
        off-robot run exit 3 -- including the eleven in each e2e chain -- for a
        robot that has nothing to stop.
        """
        from stage_runner import transitions

        robot = self._robot(with_base=False)
        outcome = transitions.stop_base(robot)
        self.assertEqual(outcome.path, transitions.STOP_PATH_PRIMARY)
        self.assertIsNone(outcome.direct_ok)
        self.assertTrue(outcome.base_is_stopped)
        self.assertEqual(outcome.error, "")
        robot.disconnect()

    def test_every_stop_path_has_a_reason_string(self) -> None:
        """``StopBaseOutcome.reason`` indexes the dict, so a gap is a KeyError."""
        from stage_runner import transitions

        paths = {
            value
            for name, value in vars(transitions).items()
            if name.startswith("STOP_PATH_")
        }
        self.assertEqual(paths, set(transitions.REASON_BY_STOP_PATH))

    def test_the_aggregators_copy_of_the_table_has_not_drifted(self) -> None:
        """aggregate.py duplicates it to stay stdlib-only; duplicates drift."""
        from stage_runner import aggregate, transitions

        self.assertEqual(
            transitions.REASON_BY_STOP_PATH,
            aggregate.STOP_BASE_REASON_BY_PATH,
            "a path aggregate does not know is filed as a FAILED stop, which is "
            "the safe direction but prints a warning on every clean run",
        )


class FiniteActionGateTest(NoRobotSdkMixin, unittest.TestCase):
    """[중요]4: no non-finite number reaches the arms.

    The premise is asserted first (``test_lerobots_own_clamp_passes_a_nan``):
    without it this whole class is guarding against nothing.
    """

    NAMES = cp.ARM_JOINT_NAMES
    MEASURED = 0.11

    def setUp(self) -> None:
        self.events: dict[str, bool] = {}

    def _gate(self):
        from stage_runner.finite_gate import FiniteActionGateStep

        return FiniteActionGateStep(self.events)

    def _action(self, **overrides):
        action: dict[str, object] = {f"{name}.pos": 0.1 for name in self.NAMES}
        action.update({"x.vel": 0.3, "theta.vel": -0.2, comp.PROGRESS_KEY: 0.5})
        action.update(overrides)
        return action

    def _observation(self):
        return {f"{name}.pos": self.MEASURED for name in self.NAMES}

    def _call(self, gate, action, observation=None):
        from lerobot.processor import create_transition

        if observation is None:
            observation = self._observation()
        return gate(create_transition(action=action, observation=observation))

    @staticmethod
    def _sent(transition):
        from lerobot.processor.core import TransitionKey

        return transition[TransitionKey.ACTION]

    def test_lerobots_own_clamp_passes_a_nan(self) -> None:
        """WHY THIS GATE EXISTS, measured against the installed lerobot.

        ``ensure_safe_goal_position`` is the 0.1 rad relative-target clamp. It is
        a min/max pair, every comparison with NaN is False, and so a NaN goal
        comes back unclamped and unflagged.
        """
        from lerobot.robots.utils import ensure_safe_goal_position

        capped = ensure_safe_goal_position(
            {"joint": (float("nan"), 0.0)}, max_relative_target=0.1
        )
        self.assertTrue(
            math.isnan(capped["joint"]),
            "if this ever starts raising or clamping, the gate's premise has "
            "changed and its docstring has to change with it",
        )
        # And a finite value 10 rad out IS clamped, so the function works.
        self.assertAlmostEqual(
            ensure_safe_goal_position({"joint": (10.0, 0.0)}, 0.1)["joint"], 0.1
        )

    def test_a_finite_action_passes_through_untouched(self) -> None:
        gate = self._gate()
        action = self._action()
        returned = self._call(gate, action)
        self.assertIs(
            self._sent(returned),
            action,
            "the common case must not even copy the dict -- it runs at 21 Hz",
        )
        self.assertFalse(gate.tripped)
        self.assertEqual(self.events, {}, "and it must not touch the event flags")

    def test_a_nan_joint_is_replaced_by_a_hold_and_breaks_the_chain(self) -> None:
        gate = self._gate()
        key = f"{self.NAMES[3]}.pos"
        action = self._action(**{key: float("nan")})
        snapshot = dict(action)
        returned = self._call(gate, action)
        sent = self._sent(returned)

        self.assertIsNot(sent, action)
        self.assertEqual(
            action,
            snapshot,
            "the dict record_loop RECORDS is this one (action_values is assigned "
            "before robot_action_processor runs), so mutating it would write a "
            "hold action the model never produced into the dataset and erase the "
            "only evidence of the fault",
        )
        for name in self.NAMES:
            self.assertEqual(sent[f"{name}.pos"], self.MEASURED)
        self.assertEqual(sent["x.vel"], 0.0)
        self.assertEqual(sent["theta.vel"], 0.0)
        self.assertEqual(sent[comp.PROGRESS_KEY], 0.0)
        self.assertEqual(
            set(sent),
            set(action),
            "SAME SHAPE: widowxai_follower.py:272 evaluates goal_pos[j] - "
            "present_pos[j] for every joint in config.joint_names, so a narrower "
            "dict is a KeyError that stops the whole write -- base included",
        )

        self.assertTrue(gate.tripped)
        self.assertEqual(gate.tripped_keys, (key,))
        self.assertTrue(self.events["exit_early"])
        self.assertTrue(self.events["stop_recording"])

    def test_an_infinite_base_velocity_trips_it_too(self) -> None:
        gate = self._gate()
        action = self._action(**{"theta.vel": float("inf")})
        sent = self._sent(self._call(gate, action))
        self.assertEqual(sent["theta.vel"], 0.0)
        self.assertEqual(gate.tripped_keys, ("theta.vel",))

    def test_a_nan_progress_slot_trips_it_although_the_arms_are_fine(self) -> None:
        """17-D only. The slot never reaches the robot, but it reaches the dataset."""
        gate = self._gate()
        action = self._action(**{comp.PROGRESS_KEY: float("nan")})
        sent = self._sent(self._call(gate, action))
        self.assertEqual(gate.tripped_keys, (comp.PROGRESS_KEY,))
        self.assertEqual(sent[comp.PROGRESS_KEY], 0.0)
        for name in self.NAMES:
            self.assertEqual(sent[f"{name}.pos"], self.MEASURED)

    def test_a_bool_is_not_a_number(self) -> None:
        """``float(True)`` is 1.0 radians, and nothing here ever means that."""
        gate = self._gate()
        action = self._action(**{f"{self.NAMES[0]}.pos": True})
        self._call(gate, action)
        self.assertTrue(gate.tripped)

    def test_a_joint_the_observation_cannot_supply_falls_back_to_the_command(
        self,
    ) -> None:
        """Only the progress slot is bad, and the observation is short one joint."""
        gate = self._gate()
        action = self._action(**{comp.PROGRESS_KEY: float("nan")})
        observation = self._observation()
        observation.pop(f"{self.NAMES[0]}.pos")
        sent = self._sent(self._call(gate, action, observation))
        self.assertEqual(
            sent[f"{self.NAMES[0]}.pos"],
            0.1,
            "repeating a finite command for a position-controlled joint is a "
            "no-op; inventing 0.0 would command it to the middle of its range",
        )

    def test_it_refuses_rather_than_invent_a_position(self) -> None:
        gate = self._gate()
        key = f"{self.NAMES[2]}.pos"
        action = self._action(**{key: float("nan")})
        observation = self._observation()
        observation.pop(key)
        with self.assertRaises(ValueError) as caught:
            self._call(gate, action, observation)
        self.assertIn(key, str(caught.exception))

    def test_a_transition_with_no_action_dict_is_left_alone(self) -> None:
        gate = self._gate()
        self.assertIsNotNone(gate({}))
        self.assertFalse(gate.tripped)


class CompletionNonFiniteTest(_MonitorHarness, unittest.TestCase):
    """[중요]5: a NaN is NO EVIDENCE, not evidence that nothing moved.

    ``max(a, nan)`` returns ``a`` and ``nan >= x`` is False, so before this a
    lost signal read as "the arm is perfectly still, exactly on an end pose",
    which is the completion condition itself.
    """

    def _raw_tick(self, monitor, *, progress=None, offset=0.0, corrupt=()):
        """One transition, with the named keys forced to NaN.

        Built here rather than through ``_tick`` because the corruption is the
        variable and ``_tick`` has no way to express it.
        """
        from lerobot.processor import create_transition

        self.now += 1 / 21.0
        action = {
            f"{name}.pos": self.params.start_pose[name] + offset for name in self.names
        }
        observation = dict(action)
        action["x.vel"] = 0.0
        action["theta.vel"] = 0.0
        if progress is not None:
            action[comp.PROGRESS_KEY] = progress
        for key in corrupt:
            target = observation if key.startswith("obs:") else action
            target[key.removeprefix("obs:")] = float("nan")
        monitor(create_transition(action=action, observation=observation))

    def _depart_raw(self, monitor, *, has_progress: bool) -> None:
        value = 0.0 if has_progress else None
        self._raw_tick(monitor, progress=value, offset=0.0)
        for _ in range(comp.DEPARTURE_ARM_TICKS):
            self._raw_tick(monitor, progress=value, offset=self.DEPARTED)

    def test_the_same_sequence_completes_when_it_is_finite(self) -> None:
        """The control. Without it the two tests below prove nothing."""
        monitor = self._monitor(has_progress=True)
        self._depart_raw(monitor, has_progress=True)
        for _ in range(40):
            self._raw_tick(monitor, progress=1.0, offset=self.DEPARTED)
            if self.events["exit_early"]:
                break
        self.assertTrue(self.events["exit_early"])
        self.assertIsNotNone(monitor.end_stage())

    def test_a_nan_measurement_is_not_a_stopped_arm(self) -> None:
        monitor = self._monitor(has_progress=True)
        self._depart_raw(monitor, has_progress=True)
        key = f"obs:{self.names[5]}.pos"
        for _ in range(40):
            self._raw_tick(
                monitor, progress=1.0, offset=self.DEPARTED, corrupt=(key,)
            )
        self.assertFalse(
            self.events["exit_early"],
            "every window containing a non-finite sample must refuse -- the "
            "peak-to-peak test would otherwise read the NaN joint as 0.0 of "
            "travel and the stage as finished",
        )
        self.assertIsNone(monitor.end_stage())

    def test_a_nan_progress_scalar_is_not_a_high_progress(self) -> None:
        """``nan < p_done`` is False, so it used to PASS the p >= p_done test."""
        monitor = self._monitor(has_progress=True)
        self._depart_raw(monitor, has_progress=True)
        for _ in range(40):
            self._raw_tick(
                monitor,
                progress=1.0,
                offset=self.DEPARTED,
                corrupt=(comp.PROGRESS_KEY,),
            )
        self.assertFalse(self.events["exit_early"])
        self.assertIsNone(monitor.end_stage())

    def test_three_consecutive_non_finite_ticks_blind_the_monitor(self) -> None:
        monitor = self._monitor(has_progress=True)
        self._depart_raw(monitor, has_progress=True)
        key = f"obs:{self.names[0]}.pos"
        for _ in range(comp.NON_FINITE_BLIND_TICKS):
            self._raw_tick(
                monitor, progress=1.0, offset=self.DEPARTED, corrupt=(key,)
            )
        self.assertTrue(monitor._stage.blind)
        # A blind monitor can never declare completion, however clean the rest of
        # the stage looks: the stage ends on its timeout or on the right arrow.
        for _ in range(40):
            self._raw_tick(monitor, progress=1.0, offset=self.DEPARTED)
        self.assertFalse(self.events["exit_early"])
        self.assertIsNone(monitor.end_stage())

    def test_one_glitched_tick_does_not_blind_it_but_is_still_not_evidence(
        self,
    ) -> None:
        monitor = self._monitor(has_progress=True)
        self._depart_raw(monitor, has_progress=True)
        key = f"obs:{self.names[0]}.pos"
        self._raw_tick(monitor, progress=1.0, offset=self.DEPARTED, corrupt=(key,))
        self.assertFalse(monitor._stage.blind, "one NaN is a glitch, not a fault")
        self.assertEqual(monitor._stage.non_finite_ticks, 1)
        self._raw_tick(monitor, progress=1.0, offset=self.DEPARTED)
        self.assertEqual(
            monitor._stage.non_finite_ticks, 0, "consecutive, not a total"
        )
        for _ in range(40):
            self._raw_tick(monitor, progress=1.0, offset=self.DEPARTED)
            if self.events["exit_early"]:
                break
        self.assertTrue(
            self.events["exit_early"],
            "it completes once the glitched sample has rolled out of the window",
        )

    def test_a_nan_never_poisons_the_base_integral_or_the_origin(self) -> None:
        monitor = self._monitor(has_progress=False)
        self._raw_tick(monitor, offset=0.0, corrupt=("x.vel",))
        stage = monitor._stage
        self.assertIsNone(
            stage.start_pose,
            "the origin is kept for the whole stage, so it must never be a NaN "
            "tick's measurement",
        )
        self.assertEqual(stage.base_fwd_m, 0.0)
        self.assertEqual(stage.base_rot_rad, 0.0)

    def test_inf_norm_refuses_a_nan_instead_of_hiding_it(self) -> None:
        pose = {name: 0.0 for name in self.names}
        measured = [0.0] * len(self.names)
        self.assertEqual(comp._inf_norm(measured, pose, self.names), 0.0)
        measured[4] = float("nan")
        self.assertIsNone(
            comp._inf_norm(measured, pose, self.names),
            "max(worst, abs(x - nan)) returns `worst`, so an unreadable joint "
            "used to vanish from the norm and a pose came out as '0.00 rad from "
            "a demonstrated end pose' -- the strongest evidence the 16-D rule "
            "accepts, produced by the absence of a measurement",
        )


class DepartureStreakTest(_MonitorHarness, unittest.TestCase):
    """[중요]7: the arm clause needs DEPARTURE_ARM_TICKS consecutive ticks."""

    def test_a_one_tick_spike_does_not_latch(self) -> None:
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.0)  # the origin
        self._tick(monitor, offset=0.5)  # one glitched encoder read
        self._tick(monitor, offset=0.0)
        self.assertFalse(
            monitor._stage.departed,
            "the latch is PERMANENT and a premise of every completion, so one "
            "tick of it hands the rest of the stage to the rule it gates",
        )
        self.assertEqual(monitor._stage.arm_departure_ticks, 0, "the streak resets")

    def test_the_required_number_of_consecutive_ticks_latches(self) -> None:
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.0)
        for index in range(comp.DEPARTURE_ARM_TICKS):
            self.assertFalse(
                monitor._stage.departed, f"not yet at {index} ticks in a row"
            )
            self._tick(monitor, offset=0.5)
        self.assertTrue(monitor._stage.departed)

    def test_the_spike_is_still_recorded_as_the_worst_distance(self) -> None:
        """Refusing to LATCH is not refusing to MEASURE."""
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.0)
        self._tick(monitor, offset=0.5)
        self._tick(monitor, offset=0.0)
        monitor.end_stage()
        self.assertAlmostEqual(monitor.last_departure.arm_rad, 0.5, places=6)
        self.assertFalse(monitor.last_departure.departed)

    def test_the_base_clause_is_not_streaked_because_it_integrates(self) -> None:
        """One spike of commanded velocity cannot cross 0.17 rad by itself."""
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.0, theta=0.0)
        self._tick(monitor, offset=0.0, theta=1.0)  # one tick = 1/21 rad
        self.assertFalse(monitor._stage.departed)
        for _ in range(4):
            self._tick(monitor, offset=0.0, theta=1.0)
        self.assertTrue(
            monitor._stage.departed,
            "integration is its own filter: five ticks at 1 rad/s is 0.24 rad",
        )


class DepartureOriginTest(_MonitorHarness, unittest.TestCase):
    """[중요]9: the origin is the stage's FIRST MEASURED pose, always.

    The case this is for is ``chain.reset.initial: false`` with
    ``from_stage > 1``: nothing has driven the arm to the designated pose, so
    measuring departure from it is either satisfied on tick 1 or unreachable.
    """

    def test_an_arm_that_starts_away_from_the_designated_pose_has_not_departed(
        self,
    ) -> None:
        monitor = self._monitor(has_progress=False)
        for _ in range(80):
            self._tick(monitor, offset=0.5)
        self.assertFalse(
            monitor._stage.departed,
            "0.5 rad from the DESIGNATED pose and 0.0 from where it started: an "
            "arm standing still has not departed, whatever the file says it "
            "should have been standing at",
        )
        self.assertFalse(self.events["exit_early"])
        monitor.end_stage()
        self.assertAlmostEqual(monitor.last_departure.arm_rad, 0.0, places=6)

    def test_it_then_departs_by_moving_from_where_it_actually_started(self) -> None:
        monitor = self._monitor(has_progress=False)
        self._tick(monitor, offset=0.5)
        for _ in range(comp.DEPARTURE_ARM_TICKS):
            self._tick(monitor, offset=0.5 + 0.2)
        self.assertTrue(monitor._stage.departed)

    def test_a_start_far_from_the_designated_pose_warns_and_nothing_else(self) -> None:
        monitor = self._monitor(has_progress=False)
        with self.assertLogs("stage_runner.completion", level="WARNING") as logs:
            self._tick(monitor, offset=comp.START_POSE_WARN_RAD + 0.2)
        self.assertTrue(
            any("did NOT start at its designated start pose" in line for line in logs.output)
        )
        monitor.end_stage()
        self.assertAlmostEqual(
            monitor.last_departure.start_pose_offset_rad,
            comp.START_POSE_WARN_RAD + 0.2,
            places=6,
        )

    def test_a_start_near_the_designated_pose_does_not_warn(self) -> None:
        monitor = self._monitor(has_progress=False)
        with self.assertNoLogs("stage_runner.completion", level="WARNING"):
            self._tick(monitor, offset=comp.START_POSE_WARN_RAD - 0.05)
        monitor.end_stage()
        self.assertLess(
            monitor.last_departure.start_pose_offset_rad, comp.START_POSE_WARN_RAD
        )


class ThresholdPreflightTest(NoRobotSdkMixin, unittest.TestCase):
    """[중요]8: every numeric knob of chain.completion / chain.reset is checked."""

    def _chain(self, **overrides):
        from stage_runner import config as cfg_module

        chain = cfg_module.ChainConfig()
        for dotted, value in overrides.items():
            block, _, field = dotted.partition(".")
            setattr(getattr(chain, block), field, value)
        return chain

    def _problems(self, **overrides) -> str:
        from stage_runner import preflight

        return "\n".join(preflight.threshold_problems(self._chain(**overrides)))

    def test_the_shipped_defaults_pass(self) -> None:
        self.assertEqual(self._problems(), "")

    def test_a_zero_stall_window_is_refused(self) -> None:
        problems = self._problems(**{"completion.stall_s": 0.0})
        self.assertIn("chain.completion.stall_s", problems)

    def test_a_nan_threshold_is_refused(self) -> None:
        """Written as ``not (x > 0)``: ``x <= 0`` is False for a NaN and PASSES."""
        for dotted in (
            "completion.stall_s",
            "completion.stall_arm_rad",
            "completion.departure_arm_rad",
            "reset.tol_rad",
            "reset.v_des_rad_s",
        ):
            with self.subTest(dotted):
                self.assertIn(
                    f"chain.{dotted}",
                    self._problems(**{dotted: float("nan")}),
                    "a NaN that passes this gate becomes a joint threshold",
                )

    def test_a_timeout_factor_of_one_is_refused(self) -> None:
        self.assertIn(
            "chain.completion.timeout_factor",
            self._problems(**{"completion.timeout_factor": 1.0}),
        )

    def test_a_zero_departure_threshold_is_refused(self) -> None:
        """At 0 the latch sets on tick 1, i.e. the latch is OFF."""
        for dotted in (
            "completion.departure_arm_rad",
            "completion.departure_base_rot_rad",
            "completion.departure_base_fwd_m",
        ):
            with self.subTest(dotted):
                self.assertIn(f"chain.{dotted}", self._problems(**{dotted: 0.0}))

    def test_the_inverted_stall_pair_is_refused(self) -> None:
        problems = self._problems(
            **{"completion.stall_track_rad": 0.02, "completion.stall_arm_rad": 0.05}
        )
        self.assertIn("TIGHTER", problems)

    def test_every_reset_knob_is_covered(self) -> None:
        for dotted in (
            "reset.t_min_s",
            "reset.v_des_rad_s",
            "reset.max_jump_rad",
            "reset.initial_max_jump_rad",
            "reset.tol_rad",
            "reset.settle_s",
        ):
            with self.subTest(dotted):
                self.assertIn(f"chain.{dotted}", self._problems(**{dotted: 0.0}))
        self.assertIn(
            "chain.reset.ceiling_factor",
            self._problems(**{"reset.ceiling_factor": 1.0}),
        )

    def test_an_initial_limit_looser_than_the_boundary_one_is_refused(self) -> None:
        self.assertIn(
            "TIGHTER",
            self._problems(
                **{"reset.initial_max_jump_rad": 2.0, "reset.max_jump_rad": 1.5}
            ),
        )

    def test_check_chain_definitions_calls_the_helper(self) -> None:
        """The helper must be wired into the gate the runner actually runs.

        A range check nothing calls is a comment. ``check_chain_definitions``
        collects every problem into ONE PreflightError, so its message is where
        the threshold has to appear; the full-config path is covered end to end by
        ``test_chain_e2e.test_a_threshold_out_of_range_exits_two``.
        """
        from stage_runner import preflight

        source = preflight.check_chain_definitions.__code__.co_names
        self.assertIn("threshold_problems", source)


class ActionWidthGateTest(NoRobotSdkMixin, unittest.TestCase):
    """[경미]10: the checkpoint's width and the dataset's must agree."""

    def _bundle(self, loaded_width: int):
        from stage_runner.policies import PolicyBundle

        class Feature:
            shape = (loaded_width,)

        config = types.SimpleNamespace(output_features={"action": Feature()})
        return PolicyBundle(
            stage_id="chain",
            policy_path="hub/model",
            config=config,
            policy=object(),
            preprocessor=object(),
            postprocessor=object(),
        )

    def _meta(self, width: int):
        return types.SimpleNamespace(
            features={
                "action": {
                    "names": [f"a{index}" for index in range(width)],
                    "shape": (width,),
                }
            }
        )

    def test_matching_widths_pass(self) -> None:
        from stage_runner import policies

        policies.check_action_width(
            self._bundle(17), dataset_meta=self._meta(17), checkpoint_action_dim=17
        )

    def test_a_seventeen_d_checkpoint_on_a_sixteen_d_dataset_is_refused(self) -> None:
        from stage_runner import policies

        with self.assertRaises(ValueError) as caught:
            policies.check_action_width(
                self._bundle(16),
                dataset_meta=self._meta(16),
                checkpoint_action_dim=17,
            )
        message = str(caught.exception)
        self.assertIn("17", message)
        self.assertIn("has_progress", message)
        self.assertIn("true", message, "it must say which value to set")

    def test_a_sixteen_d_checkpoint_on_a_seventeen_d_dataset_is_refused(self) -> None:
        from stage_runner import policies

        with self.assertRaises(ValueError) as caught:
            policies.check_action_width(
                self._bundle(17),
                dataset_meta=self._meta(17),
                checkpoint_action_dim=16,
            )
        self.assertIn("false", str(caught.exception))

    def test_a_mock_chain_has_nothing_to_compare_and_passes(self) -> None:
        from stage_runner import policies

        policies.check_action_width(
            self._bundle(16), dataset_meta=self._meta(16), checkpoint_action_dim=None
        )


class ChainPreflightScriptTest(NoRobotSdkMixin, unittest.TestCase):
    """[경미]10, the shell side: ``has_progress`` must be stated, not derived.

    ``scripts/_chain_preflight.py`` is loaded BY PATH -- it is a script, not an
    importable module, and the repo rule is that plugin and script files are
    loaded this way rather than by importing a package.
    """

    def setUp(self) -> None:
        import importlib.util
        import tempfile

        path = REPO_ROOT / "scripts" / "_chain_preflight.py"
        spec = importlib.util.spec_from_file_location("_chain_preflight_mod", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.tmp = Path(tempfile.mkdtemp(prefix="chain_preflight_"))

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def _checkpoint(
        self,
        *,
        action: int,
        state: int = 27,
        chunk: int = 30,
        env: int | None = None,
        env_type: str = "ENV",
    ) -> Path:
        directory = self.tmp / f"ckpt_{action}_{state}_{env}_{env_type}"
        directory.mkdir(parents=True, exist_ok=True)
        inputs: dict[str, dict] = {"observation.state": {"shape": [state]}}
        if env is not None:
            inputs["observation.environment_state"] = {
                "type": env_type,
                "shape": [env],
            }
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "input_features": inputs,
                    "output_features": {"action": {"shape": [action]}},
                    "chunk_size": chunk,
                }
            ),
            encoding="utf-8",
        )
        return directory

    def _yaml(self, has_progress, onehot_k: int | None = 11) -> Path:
        body = [
            "dataset:",
            "  fps: 21",
            "chain:",
            "  model:",
            f"    onehot_k: {'null' if onehot_k is None else onehot_k}",
            "    n_action_steps: 30",
        ]
        if has_progress is not None:
            body.append(f"    has_progress: {has_progress}")
        path = self.tmp / f"chain_{has_progress}_{onehot_k}.yaml"
        path.write_text("\n".join(body) + "\n", encoding="utf-8")
        return path

    def _run(
        self,
        *,
        action: int,
        has_progress,
        state: int = 27,
        env: int | None = None,
        env_type: str = "ENV",
        onehot_k: int | None = 11,
    ) -> int:
        return self.module.main(
            [
                str(
                    self._checkpoint(
                        action=action, state=state, env=env, env_type=env_type
                    )
                ),
                str(self._yaml(has_progress, onehot_k)),
                str(MOCK_PARAMS),
            ]
        )

    def test_an_explicit_true_against_a_seventeen_d_checkpoint_passes(self) -> None:
        self.assertEqual(self._run(action=17, has_progress="true"), 0)

    def test_an_explicit_false_against_a_sixteen_d_checkpoint_passes(self) -> None:
        self.assertEqual(self._run(action=16, has_progress="false"), 0)

    def test_a_null_has_progress_is_refused(self) -> None:
        """It used to mean "read it off the checkpoint", which left the one fact
        that sizes the normalizer out of the config file."""
        self.assertEqual(self._run(action=17, has_progress="null"), 3)
        self.assertEqual(self._run(action=17, has_progress=None), 3)

    def test_a_key_that_disagrees_with_the_width_is_refused(self) -> None:
        self.assertEqual(self._run(action=16, has_progress="true"), 3)
        self.assertEqual(self._run(action=17, has_progress="false"), 3)

    def test_a_round_2_env_token_checkpoint_passes(self) -> None:
        """Round 2 must get through the gate, not be rejected as "unknown"."""
        self.assertEqual(
            self._run(action=16, has_progress="false", state=27, env=11), 0
        )

    def test_an_env_width_that_is_not_k_is_refused(self) -> None:
        self.assertEqual(
            self._run(action=16, has_progress="false", state=27, env=10), 3
        )

    def test_an_env_token_without_a_onehot_k_is_refused(self) -> None:
        """The dangerous shape: a 16-D state matches the robot, so every width
        check passes and only the unfillable ENV key is wrong -- which ACT
        discovers on frame 1, after ``robot.connect()``."""
        self.assertEqual(
            self._run(
                action=16, has_progress="false", state=16, env=11, onehot_k=None
            ),
            3,
        )

    def test_an_env_feature_of_the_wrong_type_is_refused(self) -> None:
        self.assertEqual(
            self._run(
                action=16, has_progress="false", state=27, env=11, env_type="STATE"
            ),
            3,
        )

    def test_both_shipped_chain_configs_state_it(self) -> None:
        """The production YAMLs must not be the thing this change breaks."""
        import yaml as yaml_module

        for name in ("chain_m1_all11.yaml", "chain_tph_all11.yaml"):
            with self.subTest(name):
                document = yaml_module.safe_load(
                    (REPO_ROOT / "configs" / "chain" / name).read_text(
                        encoding="utf-8"
                    )
                )
                declared = document["chain"]["model"]["has_progress"]
                self.assertIsInstance(
                    declared,
                    bool,
                    f"{name} must state chain.model.has_progress explicitly; "
                    "null is refused by preflight from 2026-10-06",
                )


if __name__ == "__main__":
    unittest.main()
