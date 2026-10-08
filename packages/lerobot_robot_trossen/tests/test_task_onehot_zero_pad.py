"""``LEROBOT_TASK_ONEHOT=0/0`` -- zero-pad only (14/16-D robot state -> ``[arms, 0, 0]``, no one-hot).

Added 2026-10-08 for the 16-D SmolVLA 11-stage checkpoint (trained with ``base_state: zero``,
run with ``include_base_in_state=false``). The K>=1 path must be behaviorally unchanged.

The module is loaded by file path (not through the package ``__init__``), so this test also runs
on a machine without the robot SDK -- same pattern as ``stage_runner/tests/test_chain_unit.py``.
"""
import importlib.util
import unittest
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.core import TransitionKey

_PATH = Path(__file__).resolve().parents[1] / "src" / "lerobot_robot_trossen" / "task_onehot_patch.py"
_spec = importlib.util.spec_from_file_location("task_onehot_patch_under_test", _PATH)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


class _Cfg:
    def __init__(self, state_dim, env_dim=None):
        self.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
        if env_dim:
            self.input_features["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(env_dim,))


def _run(step, state):
    out = step({TransitionKey.OBSERVATION: {"observation.state": state}})[TransitionKey.OBSERVATION]
    return out["observation.state"], out


class ZeroPadOnlyTest(unittest.TestCase):
    def test_a_parse_accepts_only_exact_zero_zero(self):
        self.assertEqual(m._parse("0/0"), (0, 0))
        self.assertEqual(m._parse("2/11"), (2, 11))
        for bad in ("1/0", "0/11", "0/1", "0/ 0 x"):
            with self.assertRaises(RuntimeError, msg=bad):
                m._parse(bad)

    def test_a_pads_14_and_16_to_16_with_zero_base_and_no_env_key(self):
        step = m._build_step(_Cfg(16), 0, 0)
        for width in (14, 16):
            state = torch.arange(1, width + 1, dtype=torch.float32)
            step._announced = False
            with self.assertLogs(m.logger, level="INFO") as logs:
                out, obs = _run(step, state)
            self.assertTrue(any("zero-pad only (no one-hot)" in line for line in logs.output), logs.output)
            self.assertEqual(tuple(out.shape), (16,))
            self.assertTrue(torch.equal(out[:14], state[:14]))
            self.assertEqual(float(out[14:].abs().sum()), 0.0)
            self.assertNotIn("observation.environment_state", obs)

    def test_a_refuses_wide_checkpoint_env_token_and_set_stage(self):
        with self.assertRaises(RuntimeError):
            m._build_step(_Cfg(27), 0, 0)
        with self.assertRaises(RuntimeError):
            m._build_step(_Cfg(16, env_dim=11), 0, 0)
        step = m._build_step(_Cfg(16), 0, 0)
        with self.assertRaises(RuntimeError):
            step.set_stage(1)

    def test_a_k11_path_unchanged(self):
        step = m._build_step(_Cfg(27), 3, 11)
        out, _ = _run(step, torch.zeros(14))
        self.assertEqual(tuple(out.shape), (27,))
        self.assertEqual(float(out[16 + 2]), 1.0)
        self.assertEqual(float(out.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
