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


def resolve_dir(path) -> Path:
    """The checkpoint as a local directory. A hub id is resolved to its snapshot (cache
    first, then the hub). Failing to resolve is an error, never "no manifest": a relative
    checkpoint whose manifest could not be fetched must not run as a joint one."""
    d = _local_dir(path)
    if d is not None:
        return d
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors"]
    # Online first: the cache may hold only what from_pretrained fetched (config, weights),
    # not the manifest or the processor files. Offline (HF_HUB_OFFLINE=1) falls to the cache.
    try:
        return Path(snapshot_download(repo_id=str(path), allow_patterns=patterns))
    except Exception:
        pass
    try:
        return Path(snapshot_download(repo_id=str(path), allow_patterns=patterns, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(
            f"rel_eef: cannot resolve {path} to a local snapshot to read its action space ({exc!r}) "
            "-- refusing to run it unchecked"
        ) from exc


def _find_file(path, name: str) -> Path | None:
    """``name`` in the checkpoint directory or up to 4 levels above it (a training run keeps
    the manifest copy at ``<run>/``, three above ``checkpoints/<step>/pretrained_model``)."""
    d = resolve_dir(path)
    for c in (d, *d.resolve().parents[:4]):
        if (c / name).is_file():
            return c / name
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


def action_stats(path, required: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    """(mean, std) of the action from the checkpoint's postprocessor unnormalizer file.

    ``required=False`` returns None when the checkpoint has no such file or keys (older
    exports) instead of raising -- only the caller knows whether that is acceptable.
    """
    from safetensors.numpy import load_file

    d = resolve_dir(path)
    files = sorted(d.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
    if len(files) != 1:
        if not required and not files:
            return None
        raise RuntimeError(f"{d}: expected one postprocessor unnormalizer file, found {len(files)}")
    st = load_file(str(files[0]))
    if "action.mean" not in st or "action.std" not in st:
        if not required:
            return None
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

    def convert(self, rel_unnorm: np.ndarray, q_obs: np.ndarray, steps: int | None = None) -> tuple[np.ndarray, dict]:
        """(K, 16) unnormalized relative chunk + (>=14,) measured joints -> (K, 16) joint chunk.

        Only the first ``steps`` steps are solved (the ones the robot can execute,
        ``n_action_steps``); the rest repeat the last solved step. Solving stops at the
        first bad step of an arm, which -- with that arm's gripper -- then holds its last
        good command for the rest of the chunk (the measured pose / first commanded gripper
        if step 0 fails).
        """
        out = np.array(rel_unnorm, dtype=np.float64, copy=True)
        n = out.shape[0] if steps is None else max(1, min(int(steps), out.shape[0]))
        info = {"pos_err_max": 0.0, "rot_err_max": 0.0, "hold_from": {}, "solved": n}
        for side, (a0, a1) in zip(("left", "right"), eef_kin.ARM_SLICES):
            q0 = np.asarray(q_obs[a0:a1], dtype=np.float64)
            if not np.all(np.isfinite(q0)):
                raise RuntimeError(f"rel_eef: measured {side} joints are not finite: {q0}")
            grip = a1  # gripper slot right after the arm (6, 13)
            rel = out[:n, a0:a1]
            finite = np.all(np.isfinite(rel), axis=1)
            # Poses for every step at once; a non-finite row is posed from zeros and never used.
            R, P = eef_kin.absolute_pose(q0, np.where(finite[:, None], rel, 0.0))
            q = np.empty((out.shape[0], 6))
            seed = q0
            first_bad = n
            for k in range(n):
                if not finite[k]:
                    first_bad = k
                    break
                qk, ep, er = eef_kin.ik(R[k], P[k], seed)
                if (not np.all(np.isfinite(qk)) or ep > POS_TOL_M or er > ROT_TOL_RAD
                        or np.max(np.abs(qk - seed)) > STEP_JUMP_RAD):
                    first_bad = k
                    break
                info["pos_err_max"] = max(info["pos_err_max"], ep)
                info["rot_err_max"] = max(info["rot_err_max"], er)
                q[k] = qk
                seed = qk
            if first_bad < n:
                info["hold_from"][side] = first_bad
                self.holds += 1
                hold_q = q[first_bad - 1] if first_bad > 0 else q0
                hold_g = out[first_bad - 1, grip] if first_bad > 0 else out[0, grip]
                q[first_bad:] = hold_q
                out[first_bad:, grip] = hold_g
            else:
                q[n:] = q[n - 1]
                out[n:, grip] = out[n - 1, grip]
            out[:, a0:a1] = q
        return out, info

    def __call__(self, chunk, q_obs, steps: int | None = None):
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
            joint[b], info = self.convert(rel[b], q_np[b], steps)
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
        if state.shape[-1] not in (ARM_DIMS, ARM_DIMS + 2):
            # 14 = arms (include_base_in_state=false), 16 = + base. Any other width means the
            # follower also reports velocity/effort, and slots 7-13 are not the right arm.
            raise RuntimeError(f"rel_eef: robot state is {state.shape[-1]} wide, expected 14 or 16")
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
        steps = getattr(getattr(policy, "config", None), "n_action_steps", None)
        return converter(original(batch, *args, **kwargs), batch[Q_KEY], steps)

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
    """Decide from the checkpoint at ``path``; wrap ``policy`` if it is relative-EEF.

    Every checkpoint's action stats are read, so a relative one is caught both without a
    manifest and under a manifest that claims joint (a stray copy one folder up).
    """
    found = read_action_space(path)
    stats = action_stats(path, required=False)
    if stats is None:
        if found is not None and found[0] == "rel_eef":
            raise RuntimeError(f"{path}: manifest says rel_eef but the action stats cannot be read")
        if found is None and _local_dir(path) is None:
            raise RuntimeError(
                f"{path}: hub checkpoint with neither a {MANIFEST} nor readable action stats in its "
                "snapshot -- cannot tell joint from relative-EEF; download it and pass the local path"
            )
        return policy  # an older local export without processor files: joint by construction
    mean, std = stats
    relative_stats = looks_relative(mean)
    if found is None or found[0] != "rel_eef":
        if relative_stats:
            raise RuntimeError(
                f"{path}: manifest {'missing' if found is None else 'says joint'} but the action stats "
                f"look relative-EEF (arm means {np.round(mean[_arm_columns()], 3).tolist()}) -- "
                "refusing to guess the action space"
            )
        return policy
    _, state_free = found
    if not relative_stats:
        raise RuntimeError(f"{path}: manifest says rel_eef but the action stats look joint-space")
    cfg = getattr(policy, "config", None)
    ptype = getattr(cfg, "type", None)
    if ptype != "act":
        # SmolVLA's select_action bypasses predict_action_chunk (modeling_smolvla.py:342):
        # the wrap would be installed and logged but never convert anything.
        raise RuntimeError(f"{path}: rel_eef is only supported for ACT, this is {ptype!r}")
    mode = (getattr(cfg, "normalization_mapping", None) or {}).get("ACTION")
    if getattr(mode, "value", mode) != "MEAN_STD":
        raise RuntimeError(f"{path}: rel_eef conversion assumes MEAN_STD action normalization, got {mode!r}")
    wrap_policy(policy, RelEefConverter(mean, std))
    _install_stash_in_record()
    msg = (f"[rel_eef] 상대 EEF 체크포인트 -- 청크를 관절로 바꿔 보낸다 (state_free={state_free}, "
           f"앞 {getattr(cfg, 'n_action_steps', '?')} 스텝 IK) -- {path}")
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
