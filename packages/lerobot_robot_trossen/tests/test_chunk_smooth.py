"""Chunk smoothing (``LEROBOT_CHUNK_SMOOTH_TICKS``): arm joints only, centred window, ends fixed.

Pure function first, then through the executor (the smoothed chunk is what is executed
and what the seam blend sees). Same fixtures as ``test_seam_smoothing.py``.
"""
import pytest
import torch
from fake_policies import BASE_INDEX_BY_NAME, FakeChunkPolicy, IdentityPostprocessor

from lerobot_robot_trossen.chunk_execution_patch import (
    ChunkExecutionSettings,
    attach_chunk_executor,
    smooth_arm_chunk,
)

# Channels: 0, 1 arm joints · 2, 3 base · 4 gripper.
ARM_INDICES = (0, 1)
GRIPPER = 4


def jittery_chunk(steps=30, amplitude=0.012):
    """A 0.01/step ramp with an alternating +/-0.012 jitter on the arms (every step reverses
    direction); base and gripper constant. A 3-tap mean leaves +/-0.004, so the smoothed ramp
    is monotone -- the case the robot study measured (direction flips 3-5x -> demo level)."""
    t = torch.arange(steps, dtype=torch.float32)
    jitter = amplitude * (-1.0) ** t
    chunk = torch.zeros(1, steps, 5)
    chunk[0, :, 0] = 0.01 * t + jitter
    chunk[0, :, 1] = -0.01 * t - jitter
    chunk[0, :, 2] = 0.3
    chunk[0, :, 3] = -0.2
    chunk[0, :, GRIPPER] = 0.7 + jitter  # gripper jitter must survive untouched
    return chunk


def direction_flips(x):
    d = torch.diff(x)
    s = torch.sign(d)
    return int(((s[1:] * s[:-1]) < 0).sum())


def test_pure_function_reduces_flips_keeps_ends_and_leaves_base_gripper():
    chunk = jittery_chunk()
    out, change = smooth_arm_chunk(chunk, ARM_INDICES, 3)
    assert out.shape == chunk.shape
    # Ends fixed exactly.
    assert torch.equal(out[0, 0], chunk[0, 0]) and torch.equal(out[0, -1], chunk[0, -1])
    # Arms: far fewer direction flips, bounded change.
    assert direction_flips(chunk[0, :, 0]) >= 25 and direction_flips(out[0, :, 0]) == 0
    assert 0 < change <= 0.1
    # Base and gripper byte-identical.
    assert torch.equal(out[0, :, 2:], chunk[0, :, 2:])
    # Interior is the 3-tap mean.
    expected = chunk[0, 4:7, 0].mean()
    assert torch.isclose(out[0, 5, 0], expected)


def test_pure_function_identity_cases_and_odd_window():
    chunk = jittery_chunk()
    same, change = smooth_arm_chunk(chunk, ARM_INDICES, 1)
    assert same is chunk and change == 0.0
    same, _ = smooth_arm_chunk(chunk, (), 3)
    assert same is chunk
    with pytest.raises(ValueError):
        smooth_arm_chunk(chunk, ARM_INDICES, 4)
    short = chunk[:, :2]
    same, _ = smooth_arm_chunk(short, ARM_INDICES, 3)
    assert same is short


def test_window_shrinks_near_ends_so_no_step_reaches_outside():
    chunk = jittery_chunk(steps=9)
    out, _ = smooth_arm_chunk(chunk, ARM_INDICES, 5)
    # Step 1 can only reach +-1, step 2 +-2, step 3 +-2 (half window).
    assert torch.isclose(out[0, 1, 0], chunk[0, 0:3, 0].mean())
    assert torch.isclose(out[0, 2, 0], chunk[0, 0:5, 0].mean())
    assert torch.isclose(out[0, 7, 0], chunk[0, 6:9, 0].mean())


class JitterPolicy(FakeChunkPolicy):
    def __init__(self, **kwargs):
        super().__init__(action_dimension=5, **kwargs)

    def predict_action_chunk(self, batch, **kwargs):
        return jittery_chunk(self.config.chunk_size)


def _settings(**values):
    defaults = {"prefetch_ticks": 0, "prefetch_inline": False, "base_lead_ticks": 0, "execution_log_path": None}
    defaults.update(values)
    return ChunkExecutionSettings(**defaults)


def test_executor_sends_the_smoothed_arms_and_logs_the_change():
    policy = JitterPolicy(action_steps=30, chunk_size=30)
    executor, note = attach_chunk_executor(policy, _settings(smooth_ticks=3))
    assert executor is not None, note
    executor.begin_loop(BASE_INDEX_BY_NAME, IdentityPostprocessor(), None, 0, arm_indices=ARM_INDICES)
    policy.reset()
    assert "chunk smoothing 3-tick" in executor.describe()
    sent = torch.stack([policy.select_action({"tick": torch.tensor([i])})[0] for i in range(30)])
    raw = jittery_chunk(30)[0]
    # First step is as predicted; interior arms are smoothed; base and gripper untouched.
    assert torch.allclose(sent[0], raw[0])
    assert direction_flips(raw[:, 0]) >= 25 and direction_flips(sent[:, 0]) == 0
    assert torch.allclose(sent[:, 2:], raw[:, 2:])
    assert executor._swap_records and executor._swap_records[0]["arm_smooth_max"] > 0


def test_settings_reject_even_or_small_window():
    import os
    from lerobot_robot_trossen import chunk_execution_patch as m

    for bad in ("2", "1", "4"):
        os.environ[m.SMOOTH_TICKS_VARIABLE] = bad
        try:
            settings = m.read_settings()
        finally:
            os.environ.pop(m.SMOOTH_TICKS_VARIABLE, None)
        assert settings.smooth_ticks == 0
        assert any("odd" in p for p in settings.problems), settings.problems
