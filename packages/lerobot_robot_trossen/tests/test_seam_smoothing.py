"""Seam blend and RTC guidance at chunk swaps.

The blend is checked as a pure function and through the executor with a policy
whose chunks are smooth ramps that jump at every swap, the way pi0's prefetched
chunks did on the robot. RTC is checked for what the executor hands the policy
(which prefix, which delay) and, with LeRobot's own RTCProcessor, for running
inside the inference_mode that predict_action and the worker open.
"""

import time
from types import SimpleNamespace

import pytest
import torch
from fake_policies import BASE_INDEX_BY_NAME, FakeChunkPolicy, IdentityPostprocessor

from lerobot_robot_trossen import chunk_execution_patch
from lerobot_robot_trossen.chunk_execution_patch import (
    SEAM_BLEND_MAX_STEP_RAD,
    ChunkExecutionSettings,
    attach_chunk_executor,
    choose_seam_blend_ticks,
    detach_chunk_executor,
    seam_blend_offsets,
)

# Channels: 0, 1 arm joints · 2, 3 base · 4 gripper.
ARM_INDICES = (0, 1)
GRIPPER = 4


def settings(**values):
    defaults = {
        "prefetch_ticks": 0,
        "prefetch_inline": False,
        "base_lead_ticks": 0,
        "execution_log_path": None,
    }
    defaults.update(values)
    return ChunkExecutionSettings(**defaults)


def attach(policy, **values):
    executor, note = attach_chunk_executor(policy, settings(**values))
    assert executor is not None, note
    executor.begin_loop(
        BASE_INDEX_BY_NAME, IdentityPostprocessor(), None, 0, arm_indices=ARM_INDICES
    )
    policy.reset()
    return executor, note


def tick_batch(tick):
    return {"tick": torch.tensor([tick])}


class RampPolicy(FakeChunkPolicy):
    """Chunks that ramp 0.01 per step and sit ``jump`` higher on every other one."""

    def __init__(self, jump=0.5, **kwargs):
        super().__init__(action_dimension=5, **kwargs)
        self.jump = jump
        self.calls = 0

    def predict_action_chunk(self, batch, **kwargs):
        tick = int(batch["tick"].item())
        offset = self.jump * (self.calls % 2)
        self.calls += 1
        steps = torch.arange(self.config.chunk_size, dtype=torch.float32)
        chunk = torch.zeros(1, self.config.chunk_size, self.action_dimension)
        chunk[0, :, 0] = 0.01 * (tick + steps) + offset
        chunk[0, :, 1] = -0.01 * (tick + steps) - offset
        chunk[0, :, 2] = 0.3 + offset  # base x.vel
        chunk[0, :, 3] = -0.2 - offset  # base theta.vel
        chunk[0, :, GRIPPER] = offset
        return chunk


# ----- pure functions ------------------------------------------------------------


def test_offsets_start_at_the_gap_with_its_slope_and_end_at_zero():
    gap = torch.tensor([0.4, -0.2])
    slope = torch.tensor([0.01, 0.03])
    ticks = 6
    offsets = seam_blend_offsets(gap, slope, ticks)
    assert offsets.shape == (ticks, 2)
    assert torch.allclose(offsets[-1], torch.zeros(2), atol=1e-6)
    # Densely sampled, one step after the gap the offset has moved by its slope.
    fine = seam_blend_offsets(gap, slope, 600)
    assert torch.allclose(fine[0], gap + slope, atol=1e-4)


def test_window_grows_with_the_gap_and_is_capped():
    zero = torch.zeros(2)
    assert choose_seam_blend_ticks(torch.tensor([0.05, 0.0]), zero, 4) == 4
    longer = choose_seam_blend_ticks(torch.tensor([0.6, 0.0]), zero, 4)
    assert longer > 4
    path = torch.cat(
        [
            torch.tensor([[0.6, 0.0]]),
            seam_blend_offsets(torch.tensor([0.6, 0.0]), zero, longer),
        ]
    )
    assert float(path.diff(dim=0).abs().max()) <= SEAM_BLEND_MAX_STEP_RAD
    assert (
        choose_seam_blend_ticks(torch.tensor([50.0, 0.0]), zero, 4, maximum_ticks=9)
        == 9
    )


def test_a_velocity_gap_does_not_run_the_window_to_the_cap():
    # The velocity part grows with the window, so no window meets 0.1 rad/tick
    # here; the gentlest one is the shortest, not the cap.
    gap, slope = torch.zeros(2), torch.tensor([0.15, 0.0])
    ticks = choose_seam_blend_ticks(gap, slope, 4)
    assert ticks == 4
    assert float(seam_blend_offsets(gap, slope, ticks).abs().max()) < 0.2


# ----- through the executor ------------------------------------------------------


def run(policy, ticks, paced=False):
    actions = []
    for tick in range(ticks):
        actions.append(policy.select_action(tick_batch(tick))[0].clone())
        if paced:
            time.sleep(0.01)
    return torch.stack(actions)


def test_blend_removes_the_arm_jump_and_leaves_base_and_gripper_alone():
    policy = RampPolicy(action_steps=30)
    executor, _ = attach(
        policy, prefetch_ticks=5, prefetch_inline=True, seam_blend_ticks=4
    )
    blended = run(policy, 120)
    records = executor._swap_records[1:]
    assert records, "no swap after the first chunk"
    assert all(record["arm_seam_raw"] > 0.45 for record in records)
    assert all(
        record["arm_seam_sent"] < SEAM_BLEND_MAX_STEP_RAD + 0.02 for record in records
    )
    steps = blended[:, list(ARM_INDICES)].diff(dim=0).abs()
    assert float(steps.max()) < SEAM_BLEND_MAX_STEP_RAD + 0.02

    reference = RampPolicy(action_steps=30)
    attach(reference, prefetch_ticks=5, prefetch_inline=True)
    plain = run(reference, 120)
    # Base and gripper are exactly what the policy planned.
    assert torch.equal(blended[:, [2, 3, GRIPPER]], plain[:, [2, 3, GRIPPER]])
    # After each window the arms run the new chunk exactly.
    for record in records:
        end = record["tick"] + record["blend_ticks"]
        if end < len(plain):
            assert torch.allclose(
                blended[end, list(ARM_INDICES)], plain[end, list(ARM_INDICES)]
            )


def test_blend_columns_and_summary(tmp_path, caplog):
    import csv
    import logging

    log_path = tmp_path / "execution.csv"
    policy = RampPolicy(action_steps=30)
    executor, _ = attach(
        policy, prefetch_ticks=5, prefetch_inline=True, seam_blend_ticks=4
    )
    executor.execution_log = chunk_execution_patch.ExecutionLog(str(log_path))
    run(policy, 70)
    with caplog.at_level(logging.INFO):
        executor.end_loop()
    lines = [line for line in log_path.open() if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    later = [row for row in rows if row["swapped"] == "1"][1:]
    assert later and all(float(row["arm_seam_raw"]) > 0.45 for row in later)
    assert all(int(row["blend_ticks"]) >= 4 for row in later)
    assert all(row["arm_seam_joint"] in ("0", "1") for row in later)
    assert all(row["prefix_gap"] for row in later)
    assert all(row["arm_seam_raw"] == "" for row in rows if row["swapped"] != "1")
    summary = next(
        record.message
        for record in caplog.records
        if "Chunk execution summary" in record.message
    )
    for label in (
        "arm seam raw median=",
        "arm seam sent median=",
        "prefix gap median=",
        "blend ticks median=",
    ):
        assert label in summary


def test_without_the_blend_the_seam_is_only_logged():
    policy = RampPolicy(action_steps=30)
    executor, _ = attach(policy, prefetch_ticks=5, prefetch_inline=True)
    run(policy, 70)
    records = executor._swap_records[1:]
    assert records and all(record["blend_ticks"] == 0 for record in records)
    assert all(record["arm_seam_sent"] == record["arm_seam_raw"] for record in records)


# ----- RTC wiring ----------------------------------------------------------------


class FakeFlowPolicy(RampPolicy):
    """Records what the executor passes; stands in for pi0's RTC surface."""

    def __init__(self, **kwargs):
        super().__init__(jump=0.0, **kwargs)
        self.config.type = "pi0"
        self.config.rtc_config = None
        self.rtc_processor_inits = 0
        self.predict_kwargs: list[dict] = []
        self.observation_ticks_seen: list[int] = []

    def init_rtc_processor(self):
        self.rtc_processor_inits += 1

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def predict_action_chunk(self, batch, **kwargs):
        self.predict_kwargs.append(kwargs)
        self.observation_ticks_seen.append(int(batch["tick"].item()))
        return super().predict_action_chunk(batch)


def test_rtc_prefix_carries_the_blended_arms_that_were_actually_sent():
    policy = FakeFlowPolicy(action_steps=20)
    # Every other chunk jumps, far enough that the blend window (capped by the 16
    # ticks a prefetched chunk runs) reaches the next launch.
    policy.jump = 1.5
    attach(policy, prefetch_ticks=4, prefetch_inline=True, rtc=True, seam_blend_ticks=4)
    sent = run(policy, 60)
    guided = list(zip(policy.predict_kwargs[1:], policy.observation_ticks_seen[1:]))
    assert guided
    arms = list(ARM_INDICES)
    differs_from_plan = False
    for kwargs, observation_tick in guided:
        prefix = kwargs["prev_chunk_left_over"][0]
        window = sent[observation_tick : observation_tick + 4]
        assert torch.allclose(prefix[:, arms], window[:, arms])
        assert torch.allclose(prefix[:, [2, 3]], window[:, [2, 3]])  # base as planned
        planned = 0.01 * torch.arange(observation_tick, observation_tick + 4)
        differs_from_plan |= not torch.allclose(prefix[:, 0] % 0.5, planned % 0.5)
    assert differs_from_plan, "no prefix fell inside a blend window"


@pytest.mark.parametrize("inline", [True, False])
def test_rtc_steers_the_prefetch_toward_the_old_chunks_remaining_actions(inline):
    policy = FakeFlowPolicy(action_steps=20)
    executor, _ = attach(policy, prefetch_ticks=4, prefetch_inline=inline, rtc=True)
    assert policy.config.rtc_config.enabled
    assert policy.config.rtc_config.execution_horizon == 4
    assert policy.rtc_processor_inits == 1
    assert "RTC guidance" in executor.describe()
    run(policy, 60, paced=not inline)
    assert policy.predict_kwargs[0] == {}  # the first chunk has nothing to follow
    guided = policy.predict_kwargs[1:]
    assert guided
    for kwargs, observation_tick in zip(guided, policy.observation_ticks_seen[1:]):
        assert kwargs["inference_delay"] == 4 and kwargs["execution_horizon"] == 4
        prefix = kwargs["prev_chunk_left_over"]
        assert prefix.shape == (1, 4, 5)
        # The prefix is the old chunk's plan for the observation tick and the 3 after.
        expected = 0.01 * torch.arange(
            observation_tick, observation_tick + 4, dtype=torch.float32
        )
        assert torch.allclose(prefix[0, :, 0], expected, atol=1e-6)


def test_rtc_needs_the_prefetch_and_a_flow_policy_and_detaches_cleanly():
    policy = FakeFlowPolicy(action_steps=20)
    executor, note = attach_chunk_executor(policy, settings(rtc=True))
    assert executor is not None and "RTC off" in note and not executor.rtc
    assert policy.config.rtc_config is None

    act = RampPolicy(action_steps=20)
    executor, note = attach_chunk_executor(act, settings(prefetch_ticks=4, rtc=True))
    assert executor is not None and "RTC off" in note and not executor.rtc

    flow = FakeFlowPolicy(action_steps=20)
    attach_chunk_executor(
        flow, settings(prefetch_ticks=4, rtc=True, rtc_max_guidance=5.0)
    )
    assert flow.config.rtc_config.max_guidance_weight == 5.0
    detach_chunk_executor(flow)
    assert flow.config.rtc_config is None and "select_action" not in flow.__dict__


def test_detach_unfreezes_the_parameters_rtc_froze():
    policy = TinyRtcPolicy()
    attach_chunk_executor(policy, settings(prefetch_ticks=4, rtc=True))
    assert not any(parameter.requires_grad for parameter in policy.parameters())
    detach_chunk_executor(policy)
    assert all(parameter.requires_grad for parameter in policy.parameters())
    assert policy.config.rtc_config is None


# ----- settings and action names ------------------------------------------------


def test_settings_read_the_new_switches(monkeypatch):
    for variable in chunk_execution_patch.SWITCH_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LEROBOT_CHUNK_SEAM_BLEND_TICKS", "4")
    read = chunk_execution_patch.read_settings()
    assert read.seam_blend_ticks == 4 and read.wants_executor and not read.rtc

    monkeypatch.setenv("LEROBOT_CHUNK_RTC", "1")
    monkeypatch.setenv("LEROBOT_CHUNK_RTC_MAX_GUIDANCE", "-3")
    read = chunk_execution_patch.read_settings()
    assert not read.rtc and read.rtc_max_guidance is None
    assert any("LEROBOT_CHUNK_RTC does nothing" in problem for problem in read.problems)
    assert any("not positive" in problem for problem in read.problems)

    monkeypatch.setenv("LEROBOT_CHUNK_PREFETCH_TICKS", "12")
    monkeypatch.setenv("LEROBOT_CHUNK_RTC_MAX_GUIDANCE", "5")
    read = chunk_execution_patch.read_settings()
    assert read.rtc and read.rtc_max_guidance == 5.0 and not read.problems


def test_arm_joints_exclude_grippers_and_base():
    names = [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_left_carriage_joint.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_left_carriage_joint.pos",
        "x.vel",
        "theta.vel",
    ]
    dataset = SimpleNamespace(features={"action": {"names": names}})
    assert chunk_execution_patch._arm_joint_indices(dataset) == (
        *range(6),
        *range(7, 13),
    )


# ----- LeRobot's RTC under inference_mode ----------------------------------------


class TinyRtcPolicy(torch.nn.Module):
    """pi0's RTC call shape with a trainable denoiser and LeRobot's RTCProcessor."""

    def __init__(self, action_steps=20, action_dimension=5, denoise_steps=5):
        super().__init__()
        torch.manual_seed(0)
        self.denoiser = torch.nn.Linear(action_dimension, action_dimension)
        self.config = SimpleNamespace(
            n_action_steps=action_steps,
            chunk_size=action_steps,
            use_amp=False,
            temporal_ensemble_coeff=None,
            compile_model=False,
            device="cpu",
            type="pi0",
            rtc_config=None,
        )
        self.action_dimension = action_dimension
        self.denoise_steps = denoise_steps
        self.rtc_processor = None

    def init_rtc_processor(self):
        from lerobot.policies.rtc.modeling_rtc import RTCProcessor

        # As pi0's: no processor without a config.
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def reset(self):
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        feature = batch["feature"]
        x_t = torch.randn(1, self.config.chunk_size, self.action_dimension)
        dt = -1.0 / self.denoise_steps
        for step in range(self.denoise_steps):
            time_value = 1.0 + step * dt

            def denoise(x):
                return self.denoiser(x) + feature

            if self._rtc_enabled():
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    inference_delay=kwargs.get("inference_delay"),
                    time=time_value,
                    original_denoise_step_partial=denoise,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                v_t = denoise(x_t)
            x_t = x_t + dt * v_t
        return x_t


def inference_batch(tick):
    # predict_action builds the batch under inference_mode.
    with torch.inference_mode():
        return {"feature": torch.full((1, 1, 5), 0.1), "tick": torch.tensor([tick])}


def test_lerobot_rtc_fails_inside_inference_mode_as_called_by_lerobot():
    """Why the executor leaves inference_mode for guided inferences."""
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    policy = TinyRtcPolicy()
    policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=4)
    policy.init_rtc_processor()
    with (
        torch.inference_mode(),
        pytest.raises(RuntimeError, match="does not require grad"),
    ):
        policy.predict_action_chunk(
            inference_batch(0),
            inference_delay=4,
            prev_chunk_left_over=torch.zeros(1, 4, 5),
            execution_horizon=4,
        )


@pytest.mark.parametrize("inline", [True, False])
def test_lerobot_rtc_runs_in_the_executor_and_pulls_the_prefix(inline):
    def prefix_gaps(rtc):
        policy = TinyRtcPolicy()
        executor, _ = attach(policy, prefetch_ticks=4, prefetch_inline=inline, rtc=rtc)
        if rtc:
            assert not any(parameter.requires_grad for parameter in policy.parameters())
        for tick in range(60):
            with torch.inference_mode():  # as predict_action calls select_action
                policy.select_action(inference_batch(tick))
            if not inline:
                time.sleep(0.01)
        executor.end_loop()
        chunk_execution_patch.join_unfinished_inferences()
        records = executor._swap_records[1:]
        assert records
        return records

    guided = [record["prefix_gap"] for record in prefix_gaps(rtc=True)]
    unguided = [record["prefix_gap"] for record in prefix_gaps(rtc=False)]
    # Same noise in both runs; only the guidance differs.
    assert max(guided) < 0.1 * min(unguided)
