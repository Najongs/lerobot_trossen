"""rel_eef_patch: relative-EEF chunk -> joint chunk, fail-closed loading, one-hot env-only step.

Loaded by file path (with the package directory on sys.path for ``eef_kin``) so it runs
without the robot SDK the package __init__ pulls in.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_DIR = Path(__file__).resolve().parents[1] / "src/lerobot_robot_trossen"


def _load(name, filename):
    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))
    spec = importlib.util.spec_from_file_location(name, _DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rp():
    return _load("rel_eef_patch_under_test", "rel_eef_patch.py")


@pytest.fixture(scope="module")
def ek():
    return _load("eef_kin_under_test", "eef_kin.py")


def _joint_path(k=100, seed=0):
    """A smooth 16-D joint chunk inside the limits plus its start state."""
    rng = np.random.default_rng(seed)
    q0 = np.array([0.0, 1.2, 0.8, 0.2, 0.1, 0.0, 0.02, 0.0, 1.2, 0.8, -0.2, -0.1, 0.0, 0.02, 0.0, 0.0])
    t = np.linspace(0, 1, k)[:, None]
    goal = q0 + np.concatenate([rng.uniform(-0.4, 0.4, 6), [0.0], rng.uniform(-0.4, 0.4, 6), [0.0], [0.3, -0.2]])
    chunk = q0 + (goal - q0) * (3 * t**2 - 2 * t**3)
    return q0, chunk


def _relative_chunk(ek, q0, chunk):
    rel = chunk.copy()
    for a0, a1 in ek.ARM_SLICES:
        rel[:, a0:a1] = ek.relative(q0[a0:a1], chunk[:, a0:a1])
    return rel


STATS = (np.full(16, 0.01), np.full(16, 0.2))


def test_converter_round_trip(rp, ek):
    q0, chunk = _joint_path()
    rel = _relative_chunk(ek, q0, chunk)
    mean, std = STATS
    conv = rp.RelEefConverter(mean, std)
    n = torch.as_tensor((rel - mean) / std, dtype=torch.float32)[None]
    out = conv(n, torch.as_tensor(q0[:14], dtype=torch.float32)[None])
    joint = out[0].double().numpy() * std + mean
    assert np.abs(joint - chunk).max() < 2e-3  # float32 round trip + IK
    assert conv.last["hold_from"] == {} and conv.holds == 0
    # grippers and base pass through the normalization unchanged
    for c in (6, 13, 14, 15):
        assert np.allclose(joint[:, c], chunk[:, c], atol=1e-5)


def test_nonfinite_step_holds_that_arm_only(rp, ek):
    q0, chunk = _joint_path(seed=1)
    rel = _relative_chunk(ek, q0, chunk)
    rel[40, 2] = np.nan  # left arm dz
    joint, info = rp.RelEefConverter(*STATS).convert(rel, q0)
    assert info["hold_from"] == {"left": 40}
    assert np.all(joint[40:, 0:6] == joint[39, 0:6])
    assert np.abs(joint[:, 7:13] - chunk[:, 7:13]).max() < 1e-3


def test_unreachable_target_holds(rp, ek):
    q0, chunk = _joint_path(seed=2)
    rel = _relative_chunk(ek, q0, chunk)
    rel[0:, 7] += 2.0  # right arm 2 m away from step 0: IK cannot reach it
    joint, info = rp.RelEefConverter(*STATS).convert(rel, q0)
    assert info["hold_from"].get("right") == 0
    assert np.all(joint[:, 7:13] == q0[7:13])  # held at the measured pose


def test_nonfinite_measured_joints_raise(rp, ek):
    q0, chunk = _joint_path()
    rel = _relative_chunk(ek, q0, chunk)
    q0 = q0.copy()
    q0[3] = np.nan
    with pytest.raises(RuntimeError):
        rp.RelEefConverter(*STATS).convert(rel, q0)


def _ckpt(tmp_path, entries=None, mean=None):
    from safetensors.numpy import save_file

    d = tmp_path / "ck"
    d.mkdir()
    if entries is not None:
        (d / "multi_manifest.json").write_text(json.dumps(entries))
    mean = np.zeros(16, np.float32) if mean is None else mean.astype(np.float32)
    save_file({"action.mean": mean, "action.std": np.full(16, 0.2, np.float32)},
              str(d / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    return d


def test_read_action_space(rp, tmp_path):
    d = _ckpt(tmp_path, [{"action_space": "rel_eef", "state_free": True}] * 3)
    assert rp.read_action_space(d) == ("rel_eef", True)
    assert rp.read_action_space(tmp_path / "nowhere") is None or True  # hub lookup may fail offline
    (d / "multi_manifest.json").write_text(json.dumps([{"action_space": "rel_eef"}, {}]))
    with pytest.raises(RuntimeError):
        rp.read_action_space(d)


class _Policy:
    def __init__(self, chunk):
        self.chunk = chunk

    def predict_action_chunk(self, batch):
        return self.chunk.clone()


def test_load_refuses_relative_stats_without_manifest(rp, tmp_path):
    d = _ckpt(tmp_path, None, mean=np.zeros(16))
    with pytest.raises(RuntimeError, match="look relative"):
        rp.on_policy_loaded(_Policy(torch.zeros(1, 4, 16)), d)


def test_joint_checkpoint_untouched(rp, tmp_path):
    mean = np.r_[np.full(6, 1.0), 0.02, np.full(6, 1.0), 0.02, 0.0, 0.0]
    d = _ckpt(tmp_path, [{"base_state": "zero"}], mean=mean)
    pol = _Policy(torch.zeros(1, 4, 16))
    original = pol.predict_action_chunk
    assert rp.on_policy_loaded(pol, d) is pol
    assert pol.predict_action_chunk == original and not hasattr(pol, "_rel_eef_converter")


def test_rel_checkpoint_wrapped_and_needs_stash(rp, ek, tmp_path):
    d = _ckpt(tmp_path, [{"action_space": "rel_eef", "state_free": True}])
    q0, chunk = _joint_path(k=8)
    rel = _relative_chunk(ek, q0, chunk)
    pol = _Policy(torch.as_tensor(rel / 0.2, dtype=torch.float32)[None])  # mean 0, std 0.2
    rp.on_policy_loaded(pol, d)
    with pytest.raises(RuntimeError, match="rel_eef_q"):
        pol.predict_action_chunk({"observation.state": torch.zeros(1, 14)})
    out = pol.predict_action_chunk({rp.Q_KEY: torch.as_tensor(q0[:14], dtype=torch.float32)[None]})
    assert np.abs(out[0].double().numpy() * 0.2 - chunk).max() < 2e-3
    with pytest.raises(RuntimeError, match="RTC"):
        pol.predict_action_chunk({rp.Q_KEY: torch.zeros(1, 14)}, prev_chunk_left_over=torch.zeros(1, 2, 16))


def test_stash_step_copies_raw_arms(rp):
    from lerobot.processor.core import TransitionKey

    state = torch.arange(16, dtype=torch.float32)[None]
    tr = {TransitionKey.OBSERVATION: {"observation.state": state}}
    out = rp.RelEefStashStep()(tr)
    q = out[TransitionKey.OBSERVATION][rp.Q_KEY]
    assert torch.equal(q, state[:, :14]) and q.data_ptr() != state.data_ptr()
    assert torch.equal(out[TransitionKey.OBSERVATION]["observation.state"], state)


def test_onehot_env_only_for_state_free():
    tp = _load("task_onehot_under_test", "task_onehot_patch.py")
    from lerobot.processor.core import TransitionKey

    class Cfg:
        input_features = {"observation.environment_state": type("F", (), {"type": "ENV", "shape": (11,)})()}

    step = tp._build_step(Cfg(), 3, 11)
    state = torch.ones(1, 14)
    out = step({TransitionKey.OBSERVATION: {"observation.state": state}})[TransitionKey.OBSERVATION]
    assert torch.equal(out["observation.state"], state)
    env = out["observation.environment_state"]
    assert env.shape == (1, 11) and env[0, 2] == 1 and env.sum() == 1
    with pytest.raises(RuntimeError):
        tp._build_step(Cfg(), 3, 10)  # K must match the env token
