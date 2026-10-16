"""Run relative-EEF ACT checkpoints (trossen-ai-simulation J57/J58) as joint-space ACT.

Why this exists
---------------
``train_multi.py`` ``action_space: rel_eef`` trains the 12 arm slots of the action as the
end-effector displacement relative to the chunk's start pose, ``[dp (3), rotvec (3)]`` per
arm in the start end-effector frame (``eef_kin.relative``), not joint angles. Sent as they
are, those numbers would be read as joint targets. This module converts every predicted
chunk back to absolute joint angles right where the policy returns it:

    policy.predict_action_chunk(batch)   normalized relative chunk (B, K, 16)
      -> unnormalize with the checkpoint's own action stats
      -> per arm: T0 = FK(measured joints at the query), targets T0 * dT_k, IK seeded
         step by step from the measured joints (eef_kin.chunk_to_joints)
      -> normalize the joint chunk with the SAME stats and return it

so the postprocessor (that checkpoint's unnormalizer, unchanged) turns it into joint
radians, and everything after it -- chunk executor, seam blend, 3-tick smoothing, the
0.1 rad clamp, pacing, the base path -- sees an ordinary absolute-joint ACT chunk.
Grippers and the base velocities were trained absolute and pass through untouched.

Fail closed
-----------
- The checkpoint declares its action space in ``multi_manifest.json`` (copied next to the
  weights by ``hub_upload_ckpt.py``). A checkpoint without the manifest whose action stats
  look relative (every arm-slot mean near 0 -- joint_1 never averages below 0.4 rad in
  joint space) is refused at load.
- The measured joints come from a preprocessor step this module inserts first
  (``observation.rel_eef_q``, a copy of the raw robot state taken before the one-hot and
  the normalizer). If the batch reaches the policy without it (a caller that built its own
  processors, e.g. the chain runner), inference raises -- before any action is sent.
- An IK step whose residual exceeds 5 mm / 0.05 rad, a non-finite value, or a step that
  jumps more than 0.3 rad from the previous one holds that arm at its last good target
  for the rest of the chunk (the measured pose if the first step fails), and is logged.
- RTC-style guided inference (a normalized-joint prefix fed back into the policy) is
  refused: the prefix would be in the wrong space.

Off for every other checkpoint: the load hook only reads ``multi_manifest.json`` and the
action stats; nothing is wrapped unless the checkpoint is ``rel_eef``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

try:
    from . import eef_kin
except ImportError:  # loaded by file path (tests), with this directory on sys.path
    import eef_kin

from lerobot.processor.pipeline import ProcessorStep

logger = logging.getLogger(__name__)

MANIFEST = "multi_manifest.json"
Q_KEY = "observation.rel_eef_q"
_STATE_KEY = "observation.state"
ARM_DIMS = 14
ACTION_DIM = 16
POS_TOL_M = 5e-3
ROT_TOL_RAD = 0.05
STEP_JUMP_RAD = 0.3
# Joint-space arm-slot means are >= 0.4 rad for joint_1 (data 0.4..2.66); relative ones are ~0.
_REL_MEAN_MAX = 0.2


# ----- reading the checkpoint ---------------------------------------------------------

def _local_dir(path) -> Path | None:
    p = Path(str(path))
    return p if p.is_dir() else None


def _find_file(path, name: str) -> Path | None:
    d = _local_dir(path)
    if d is not None:
        for c in (d / name, d.parent / name):
            if c.is_file():
                return c
        return None
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=str(path), filename=name))
    except Exception:
        return None


def read_action_space(path) -> tuple[str, bool] | None:
    """(action_space, state_free) from the checkpoint's manifest copy; None if there is none."""
    man = _find_file(path, MANIFEST)
    if man is None:
        return None
    data = json.loads(man.read_text())
    entries = data if isinstance(data, list) else data.get("entries", data)
    spaces = {e.get("action_space", "joint") for e in entries}
    free = {bool(e.get("state_free", False)) for e in entries}
    if len(spaces) != 1 or len(free) != 1:
        raise RuntimeError(
            f"{man}: action_space {sorted(spaces)} / state_free {sorted(free)} differ between "
            "entries -- the robot cannot run one checkpoint in two action spaces"
        )
    space = spaces.pop()
    if space not in ("joint", "rel_eef"):
        raise RuntimeError(f"{man}: unknown action_space {space!r}")
    return space, free.pop()


def action_stats(path) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) of the action from the checkpoint's postprocessor unnormalizer file."""
    from safetensors.numpy import load_file

    d = _local_dir(path)
    if d is None:
        from huggingface_hub import snapshot_download

        d = Path(snapshot_download(repo_id=str(path), allow_patterns=["*.safetensors", "*.json"]))
    files = sorted(d.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
    if len(files) != 1:
        raise RuntimeError(f"{d}: expected one postprocessor unnormalizer file, found {len(files)}")
    st = load_file(str(files[0]))
    if "action.mean" not in st or "action.std" not in st:
        raise RuntimeError(f"{files[0]}: no action.mean/action.std (keys {sorted(st)[:8]})")
    mean = np.asarray(st["action.mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(st["action.std"], dtype=np.float64).reshape(-1)
    return mean, std


def _arm_columns() -> list[int]:
    return [i for a0, a1 in eef_kin.ARM_SLICES for i in range(a0, a1)]


def looks_relative(mean: np.ndarray) -> bool:
    if mean.shape[0] < ACTION_DIM:
        return False
    return bool(np.all(np.abs(mean[_arm_columns()]) < _REL_MEAN_MAX))


# ----- conversion ---------------------------------------------------------------------

class RelEefConverter:
    """Normalized relative chunk (B, K, 16) + measured joints (B, >=14) -> normalized joint chunk."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        if mean.shape != (ACTION_DIM,) or std.shape != (ACTION_DIM,):
            raise RuntimeError(f"rel_eef: action stats must be {ACTION_DIM}-D, got {mean.shape}/{std.shape}")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
            raise RuntimeError("rel_eef: action stats are not finite/positive")
        self.mean = mean
        self.std = std
        self.chunks = 0
        self.holds = 0
        self.last = None  # summary of the last chunk, for logs/tests

    def convert(self, rel_unnorm: np.ndarray, q_obs: np.ndarray) -> tuple[np.ndarray, dict]:
        """(K, 16) unnormalized relative chunk + (>=14,) measured joints -> (K, 16) joint chunk."""
        out = np.array(rel_unnorm, dtype=np.float64, copy=True)
        info = {"pos_err_max": 0.0, "rot_err_max": 0.0, "hold_from": {}}
        for side, (a0, a1) in zip(("left", "right"), eef_kin.ARM_SLICES):
            q0 = np.asarray(q_obs[a0:a1], dtype=np.float64)
            if not np.all(np.isfinite(q0)):
                raise RuntimeError(f"rel_eef: measured {side} joints are not finite: {q0}")
            rel = out[:, a0:a1]
            ok_rel = np.all(np.isfinite(rel), axis=1)
            safe_rel = np.where(ok_rel[:, None], rel, 0.0)
            q, ep, er = eef_kin.chunk_to_joints(q0, safe_rel)
            prev = np.vstack([q0[None], q[:-1]])
            bad = (~ok_rel) | (~np.all(np.isfinite(q), axis=1)) | (ep > POS_TOL_M) | (er > ROT_TOL_RAD) \
                | (np.max(np.abs(q - prev), axis=1) > STEP_JUMP_RAD)
            first_bad = int(np.argmax(bad)) if bad.any() else len(bad)
            if first_bad < len(bad):
                q[first_bad:] = q[first_bad - 1] if first_bad > 0 else q0
                info["hold_from"][side] = first_bad
                self.holds += 1
            good = np.arange(len(bad)) < first_bad
            if good.any():
                info["pos_err_max"] = max(info["pos_err_max"], float(ep[good].max()))
                info["rot_err_max"] = max(info["rot_err_max"], float(er[good].max()))
            out[:, a0:a1] = q
        return out, info

    def __call__(self, chunk, q_obs):
        import torch

        if chunk.ndim != 3 or chunk.shape[-1] != ACTION_DIM:
            raise RuntimeError(f"rel_eef: expected a (B, K, {ACTION_DIM}) chunk, got {tuple(chunk.shape)}")
        q_np = q_obs.detach().to("cpu", torch.float64).numpy().reshape(chunk.shape[0], -1)
        if q_np.shape[1] < ARM_DIMS:
            raise RuntimeError(f"rel_eef: measured state is {q_np.shape[1]} wide, need >= {ARM_DIMS}")
        x = chunk.detach().to("cpu", torch.float64).numpy()
        rel = x * self.std + self.mean
        joint = np.empty_like(rel)
        for b in range(rel.shape[0]):
            joint[b], info = self.convert(rel[b], q_np[b])
            self.last = info
            if info["hold_from"]:
                logger.warning(
                    f"[rel_eef] IK 실패/튐 -- 팔을 그 스텝부터 마지막 성공 자세로 유지: {info['hold_from']} "
                    f"(잔차 한도 {POS_TOL_M * 1e3:.0f} mm / {ROT_TOL_RAD} rad, 스텝 튐 {STEP_JUMP_RAD} rad)"
                )
        self.chunks += 1
        if self.chunks == 1 or self.chunks % 20 == 0:
            logger.info(
                f"[rel_eef] 청크 {self.chunks}: IK 잔차 최대 {self.last['pos_err_max'] * 1e3:.2f} mm / "
                f"{self.last['rot_err_max']:.4f} rad · 유지 누적 {self.holds}"
            )
        normalized = (joint - self.mean) / self.std
        return torch.as_tensor(normalized, dtype=chunk.dtype, device=chunk.device)


# ----- hooks ----------------------------------------------------------------------------

class RelEefStashStep(ProcessorStep):
    """Preprocessor step (inserted first): copy the raw robot state to ``observation.rel_eef_q``.

    First, so it sees the robot's own joint radians -- before the one-hot step rebuilds the
    state and before the normalizer. The key is not a policy feature, so the normalizer
    skips it and ACT never reads it; only the wrapped ``predict_action_chunk`` does.
    """

    def __call__(self, transition):
        from lerobot.processor.core import TransitionKey

        obs = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(obs, dict) or _STATE_KEY not in obs:
            raise RuntimeError(
                f"rel_eef: the observation has no {_STATE_KEY} -- the relative chunk needs the "
                "measured joints"
            )
        state = obs[_STATE_KEY]
        if state.shape[-1] < ARM_DIMS:
            raise RuntimeError(f"rel_eef: robot state is {state.shape[-1]} wide, need >= {ARM_DIMS}")
        obs = dict(obs)
        obs[Q_KEY] = state[..., :ARM_DIMS].clone()
        transition = transition.copy()
        transition[TransitionKey.OBSERVATION] = obs
        return transition

    def get_config(self):
        return {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def reset(self):
        pass

    def transform_features(self, features):
        return features


def wrap_policy(policy, converter: RelEefConverter):
    original = policy.predict_action_chunk

    def predict_action_chunk(batch, *args, **kwargs):
        if kwargs.get("prev_chunk_left_over") is not None:
            raise RuntimeError("rel_eef: guided (RTC) inference would feed a joint-space prefix back in")
        if Q_KEY not in batch:
            raise RuntimeError(
                f"rel_eef: {Q_KEY} is missing from the batch -- the preprocessor was not built "
                "through lerobot-record's make_pre_post_processors. Refusing to send relative "
                "end-effector actions as joint targets"
            )
        return converter(original(batch, *args, **kwargs), batch[Q_KEY])

    policy.predict_action_chunk = predict_action_chunk
    policy._rel_eef_converter = converter
    return policy


def _install_stash_in_record() -> None:
    """Wrap lerobot_record's make_pre_post_processors (already imported: we are inside record())."""
    module = sys.modules.get("lerobot.scripts.lerobot_record")
    if module is None:
        return  # not lerobot-record; predict_action_chunk will refuse without the key
    original = module.make_pre_post_processors
    if getattr(original, "_rel_eef_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        pre, post = original(*args, **kwargs)
        steps = list(pre.steps)
        steps.insert(0, RelEefStashStep())
        pre.steps = steps
        logger.warning("[rel_eef] 전처리 맨 앞에 측정 관절 복사 단계(observation.rel_eef_q)를 넣었다")
        return pre, post

    wrapped._rel_eef_wrapped = True
    wrapped.__wrapped__ = original
    module.make_pre_post_processors = wrapped


def on_policy_loaded(policy, path):
    """Decide from the checkpoint at ``path``; wrap ``policy`` if it is relative-EEF."""
    found = read_action_space(path)
    if found is None:
        mean, _ = action_stats(path) if _local_dir(path) is not None else (None, None)
        if mean is not None and looks_relative(mean):
            raise RuntimeError(
                f"{path}: no {MANIFEST} but the action stats look relative-EEF (arm means "
                f"{np.round(mean[_arm_columns()], 3).tolist()}) -- refusing to guess the action space"
            )
        return policy
    space, state_free = found
    if space != "rel_eef":
        return policy
    mean, std = action_stats(path)
    if not looks_relative(mean):
        raise RuntimeError(f"{path}: manifest says rel_eef but the action stats look joint-space")
    wrap_policy(policy, RelEefConverter(mean, std))
    _install_stash_in_record()
    msg = (f"[rel_eef] 상대 EEF 체크포인트 -- 청크를 관절로 바꿔 보낸다 (state_free={state_free}) -- {path}")
    print(msg, flush=True)
    logger.warning(msg)
    return policy


def install_load_hook(policy_base=None) -> None:
    """Wrap ``PreTrainedPolicy.from_pretrained``; idempotent."""
    if policy_base is None:
        from lerobot.policies.pretrained import PreTrainedPolicy as policy_base
    if getattr(policy_base, "_rel_eef_hooked", False):
        return
    original = policy_base.from_pretrained.__func__

    def from_pretrained(cls, pretrained_name_or_path, *args, **kwargs):
        policy = original(cls, pretrained_name_or_path, *args, **kwargs)
        return on_policy_loaded(policy, pretrained_name_or_path)

    policy_base.from_pretrained = classmethod(from_pretrained)
    policy_base._rel_eef_hooked = True
