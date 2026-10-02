"""Policies and a robot for the base latency tests; no hardware, no network."""

import threading
import time
from types import SimpleNamespace

import torch

ACTION_NAMES = ["left_joint_0.pos", "right_joint_0.pos", "x.vel", "theta.vel"]
BASE_INDEX_BY_NAME = {"x.vel": 2, "theta.vel": 3}


def encode(observation_tick: int, step: int, channel: int) -> float:
    return float(observation_tick * 1000 + step * 10 + channel)


def decode(value: float) -> tuple[int, int, int]:
    value = int(round(float(value)))
    return value // 1000, (value % 1000) // 10, value % 10


class FakeChunkPolicy:
    """Chunks whose values encode (observation tick, chunk step, channel).

    The observation tick comes from the batch (``batch["tick"]``), so every
    action says exactly which observation and which chunk step produced it.
    """

    def __init__(
        self,
        action_steps: int = 10,
        chunk_size: int | None = None,
        action_dimension: int = 4,
        delay_seconds: float = 0.0,
        slow_calls: tuple[int, ...] = (),
        failing_calls: tuple[int, ...] = (),
    ):
        self.config = SimpleNamespace(
            n_action_steps=action_steps,
            chunk_size=chunk_size or action_steps,
            use_amp=False,
            temporal_ensemble_coeff=None,
            compile_model=False,
            device="cpu",
            # Stands in for ACT's select_action semantics (a plain chunk queue),
            # which is what the executor's type allowlist checks for.
            type="act",
        )
        self.action_dimension = action_dimension
        self.delay_seconds = delay_seconds
        self.slow_calls = slow_calls
        self.failing_calls = failing_calls
        self.observation_ticks: list[int] = []
        self.reset_count = 0
        self.active_inferences = 0
        self.most_concurrent_inferences = 0
        self._lock = threading.Lock()
        self._call_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, batch):
        raise AssertionError("the chunk executor must replace select_action")

    def predict_action_chunk(self, batch):
        with self._lock:
            call_index = self._call_count
            self._call_count += 1
            self.active_inferences += 1
            self.most_concurrent_inferences = max(
                self.most_concurrent_inferences, self.active_inferences
            )
        try:
            delay = self.delay_seconds
            if call_index in self.slow_calls:
                delay = max(delay, 1.0)
            if delay:
                time.sleep(delay)
            if call_index in self.failing_calls:
                raise RuntimeError(f"injected failure on call {call_index}")
            observation_tick = int(batch["tick"].item())
            self.observation_ticks.append(observation_tick)
            steps = torch.arange(self.config.chunk_size).view(1, -1, 1)
            channels = torch.arange(self.action_dimension).view(1, 1, -1)
            return (observation_tick * 1000 + steps * 10 + channels).to(torch.float32)
        finally:
            with self._lock:
                self.active_inferences -= 1


class QueuePolicy(FakeChunkPolicy):
    """A FakeChunkPolicy with lerobot's own select_action queue logic."""

    def reset(self) -> None:
        super().reset()
        self._queue = []

    def select_action(self, batch):
        if not getattr(self, "_queue", None):
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._queue = list(chunk.transpose(0, 1))
        return self._queue.pop(0)


def tick_batch(tick: int) -> dict:
    return {"tick": torch.tensor([tick])}


def tiny_act_policy(seed: int = 0):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    config = ACTConfig(
        device="cpu",
        chunk_size=10,
        n_action_steps=10,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.environment_state": PolicyFeature(
                type=FeatureType.ENV, shape=(3,)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
        },
    )
    torch.manual_seed(seed)
    policy = ACTPolicy(config)
    policy.eval()
    return policy


class FakeRobot:
    """Just enough of a robot for lerobot's real record_loop."""

    name = "fake_robot"
    robot_type = "fake_robot"

    def __init__(self):
        self.sent_actions: list[dict] = []
        self.base = SimpleNamespace(set_cmd_vel=self._set_cmd_vel)
        self.base_stops = 0

    def _set_cmd_vel(self, x_velocity: float, theta_velocity: float) -> bool:
        self.base_stops += 1
        return True

    def get_observation(self) -> dict:
        return {"tick_source": float(len(self.sent_actions))}

    def send_action(self, action: dict) -> dict:
        self.sent_actions.append(dict(action))
        return action


class FakeDataset:
    """The dataset surface record_loop and the switches touch."""

    def __init__(self, fps: int, repository_id: str = "local/eval_fake"):
        self.fps = fps
        self.repo_id = repository_id
        self.num_episodes = 0
        self.frames: list[dict] = []
        self.features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["tick_source"],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(ACTION_NAMES),),
                "names": list(ACTION_NAMES),
            },
        }

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)


class TickPreprocessor:
    """Turns the observation into the tick batch the fake policies read."""

    def __init__(self):
        self.tick = 0

    def __call__(self, observation: dict) -> dict:
        batch = {"tick": torch.tensor([self.tick])}
        self.tick += 1
        return batch

    def reset(self) -> None:
        self.tick = 0


class IdentityPostprocessor:
    def __call__(self, action):
        return action.to("cpu")

    def reset(self) -> None:
        pass
