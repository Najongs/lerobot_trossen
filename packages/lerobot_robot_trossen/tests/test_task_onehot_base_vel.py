"""``LEROBOT_TASK_ONEHOT_BASE=vel`` -- feed measured base velocity into state slots 14-15 (J45, 2026-10-14).

A checkpoint trained with ``base_state: vel`` must see ``[arms, x.vel, theta.vel, one_hot]``; every other
checkpoint keeps the zero slots. Loaded by file path so it runs without the robot SDK (as test_task_onehot_zero_pad).
"""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor.core import TransitionKey

_PATH = Path(__file__).resolve().parents[1] / "src" / "lerobot_robot_trossen" / "task_onehot_patch.py"
_spec = importlib.util.spec_from_file_location("task_onehot_patch_base_vel", _PATH)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


class _Cfg:
    def __init__(self, state_dim):
        self.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}


def _run(step, state):
    return step({TransitionKey.OBSERVATION: {"observation.state": state}})[TransitionKey.OBSERVATION]["observation.state"]


class BaseVelTest(unittest.TestCase):
    def test_default_and_zero_keep_zero_slots(self):
        state = torch.arange(1, 17, dtype=torch.float32)
        for val in (None, "zero", ""):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LEROBOT_TASK_ONEHOT_BASE", None)
                if val is not None:
                    os.environ["LEROBOT_TASK_ONEHOT_BASE"] = val
                step = m._build_step(_Cfg(27), 2, 11)
            self.assertEqual(step.base_mode, "zero")
            out = _run(step, state)
            self.assertTrue(torch.equal(out[:14], state[:14]))
            self.assertEqual(float(out[14:16].abs().sum()), 0.0)
            self.assertEqual(float(out[16 + 1]), 1.0)

    def test_vel_copies_measured_slots(self):
        state = torch.arange(1, 17, dtype=torch.float32)
        with mock.patch.dict(os.environ, {"LEROBOT_TASK_ONEHOT_BASE": "vel"}):
            step = m._build_step(_Cfg(27), 3, 11)
        with self.assertLogs(m.logger, level="INFO") as logs:
            out = _run(step, state)
        self.assertTrue(any("base slots = measured velocity" in l for l in logs.output), logs.output)
        self.assertEqual(tuple(out.shape), (27,))
        self.assertTrue(torch.equal(out[:16], state[:16]))
        self.assertEqual(float(out[16 + 2]), 1.0)
        self.assertEqual(float(out[16:].sum()), 1.0)
        # the robot tensor is not aliased: an in-place edit downstream must not reach the observation
        out[14] = 99.0
        self.assertEqual(float(state[14]), 15.0)

    def test_vel_refuses_14d_robot_state(self):
        with mock.patch.dict(os.environ, {"LEROBOT_TASK_ONEHOT_BASE": "vel"}):
            step = m._build_step(_Cfg(27), 1, 11)
        with self.assertRaises(RuntimeError):
            _run(step, torch.zeros(14))

    def test_vel_refuses_zero_pad_only_and_bad_value(self):
        with mock.patch.dict(os.environ, {"LEROBOT_TASK_ONEHOT_BASE": "vel"}):
            with self.assertRaises(RuntimeError):
                m._build_step(_Cfg(16), 0, 0)
        with mock.patch.dict(os.environ, {"LEROBOT_TASK_ONEHOT_BASE": "odom"}):
            with self.assertRaises(RuntimeError):
                m._build_step(_Cfg(27), 1, 11)


if __name__ == "__main__":
    unittest.main()
