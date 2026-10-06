"""Start-pose guide for multi-stage eval resets (off by default, opt in per run).

Why this exists
---------------
The multi-stage ACT reads the stage mostly from the scene and the arms' start pose,
not from the one-hot (docs/eval_najy_results_1006.md, offline check). Setting the
start pose by eye with the leader arms is hard, so during the reset phase this logs,
once a second, how far each arm joint is from that stage's training start pose.

Logging only: it reads the action already sent and never changes it. During the
policy phase it is silent except for one ``POSE-START`` line on the first tick of
each episode (episode 0 has no reset before it, so this is the only record of its
start pose in the log; the value is the first *action* sent, which the relative
clamp keeps within ``max_relative_target`` of the measured pose).

Usage
-----
    LEROBOT_POSE_GUIDE=5 uv run lerobot-record ...      # task05 target (1-based)

``scripts/eval_najy.sh`` sets it from the stage number (``POSE_GUIDE=0`` turns it off).

Targets: per-joint median of every training episode's first frame, degrees
(docs/eval_najy.md "단계별 학습 시작 자세"). task08 and task10 start from several
distinct poses, so their median may be a pose no demo used.
"""

import logging
import math
import os
import time

from lerobot_robot_trossen.loop_rate_log import current_phase, current_phase_seq

logger = logging.getLogger(__name__)

_ENV_VAR = "LEROBOT_POSE_GUIDE"
_PERIOD_S = 1.0

# stage -> (left j0..j5, right j0..j5), degrees
TARGETS_DEG = {
    1: ((0, 60, 35, 27, 0, 0), (0, 60, 32, 28, 0, 0)),
    2: ((-13, 78, 28, 36, 5, 7), (12, 74, 28, 37, -14, -3)),
    3: ((-21, 92, 22, 42, 41, 23), (19, 84, 24, 29, -44, -22)),
    4: ((-21, 84, 26, 27, 24, 15), (30, 80, 29, 19, -5, -9)),
    5: ((-14, 90, 30, 49, 0, 0), (13, 91, 30, 47, -15, -2)),
    6: ((-19, 83, 17, 44, 15, 6), (20, 90, 28, 36, -13, -2)),
    7: ((-25, 89, 27, 42, 24, 10), (18, 78, 19, 49, -8, 0)),
    8: ((-36, 97, 14, 75, -31, -3), (29, 75, 4, 66, 27, 8)),
    9: ((-47, 81, 3, 74, -42, -1), (48, 75, 6, 63, 42, 6)),
    10: ((-32, 79, 18, 45, -2, 0), (30, 69, 18, 50, 6, 0)),
    11: ((-22, 93, 34, 52, 11, 6), (22, 94, 33, 53, -7, -3)),
}
_MULTIMODAL = {8, 10}


def _parse() -> int | None:
    raw = os.getenv(_ENV_VAR, "").strip()
    if not raw or raw in ("0", "off", "false", "no"):
        return None
    try:
        stage = int(raw.removeprefix("task"))
    except ValueError:
        logger.warning(f"{_ENV_VAR}={raw!r}: expected a stage number 1-11; guide off.")
        return None
    if stage not in TARGETS_DEG:
        logger.warning(f"{_ENV_VAR}={raw!r}: stage must be 1-11; guide off.")
        return None
    return stage


_STAGE = _parse()
_last_t = 0.0
_prev_seq = -1  # phase generation of the last tick seen (loop_rate_log.current_phase_seq)
# Overrides TARGETS_DEG[_STAGE] when the caller knows the target better than this
# table does -- see set_stage().
_TARGET_OVERRIDE: tuple[tuple[float, ...], tuple[float, ...]] | None = None
# Phases that get the once-a-second "how far to the target" line. "teleop" is the
# leader-arm reset of `lerobot-record`; "reset" is the chain runner's automatic
# pose ramp, which is the one phase where this readout can be checked against a
# target the runner itself is driving towards.
_GUIDE_PHASES = ("teleop", "reset")


def set_stage(
    stage: int | None,
    target_deg: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> None:
    """Re-point the guide at another stage inside ONE process (chain runner).

    ``lerobot-record`` runs one stage per process, so the stage came from the
    environment once at import. A chain runs all 11 in a single process and the
    reset before stage k must be measured against stage k's target, not stage
    1's.

    ``target_deg`` replaces :data:`TARGETS_DEG` for this stage, as
    ``((left j0..j5), (right j0..j5))`` in DEGREES. Pass it whenever the caller
    has the DESIGNATED reset pose: this module's table is the per-joint MEDIAN of
    every demo's first frame, and for t02·03·04·05·10 the start poses are
    bimodal, so the median is a pose no demonstration ever used
    (docs/eval_najy.md, offline §92.1). Measuring a ramp whose target came from
    `configs/chain/stage_params.json` against the median would print a nonzero
    distance at the exact moment the ramp has arrived. ``None`` restores the
    table.

    ``stage=None`` turns the guide off, which is what a run with no stage context
    (an initial reset before stage 1 has been chosen) should print: nothing.
    """
    global _STAGE, _TARGET_OVERRIDE, _last_t
    if stage is not None and stage not in TARGETS_DEG and target_deg is None:
        logger.warning(f"{_ENV_VAR}: stage {stage} has no target; guide off.")
        stage = None
    if target_deg is not None and (
        len(target_deg) != 2 or any(len(side) != 6 for side in target_deg)
    ):
        raise ValueError(
            "pose_guide target_deg must be ((left j0..j5), (right j0..j5)) in degrees"
        )
    _STAGE = stage
    _TARGET_OVERRIDE = target_deg
    # So the first tick of the new stage prints immediately instead of waiting
    # out the remainder of the previous stage's one-second period.
    _last_t = 0.0


def _target() -> tuple[tuple[float, ...], tuple[float, ...]]:
    if _TARGET_OVERRIDE is not None:
        return _TARGET_OVERRIDE
    return TARGETS_DEG[_STAGE]


def pose_guide_tick(sent_arms: dict) -> None:
    """Called once per loop with the arm action just sent.

    Reset phase (leader-arm ``teleop`` or the chain runner's ``reset``): one
    ``POSE`` line per second. Policy phase: one ``POSE-START`` line on the first
    tick of each episode, then silence.
    """
    global _last_t, _prev_seq
    if _STAGE is None:
        return
    phase = current_phase()
    seq = current_phase_seq()
    # A new record_loop call (episode or reset) since the last tick -- robust to a reset that
    # ran for zero ticks, where the phase string alone would stay "policy".
    first_policy_tick = phase == "policy" and seq != _prev_seq
    _prev_seq = seq
    if phase in _GUIDE_PHASES:
        now = time.monotonic()
        if now - _last_t < _PERIOD_S:
            return
        _last_t = now
        tag = "POSE"
    elif first_policy_tick:
        tag = "POSE-START"
    else:
        return
    try:
        parts, sq, worst = [], 0.0, (0.0, "")
        for side, target in zip(("left", "right"), _target()):
            cells = []
            for j, tgt in enumerate(target):
                cur = math.degrees(sent_arms[f"{side}_joint_{j}.pos"])
                need = tgt - cur
                sq += math.radians(need) ** 2
                if abs(need) > abs(worst[0]):
                    worst = (need, f"{side[0].upper()} j{j}")
                mark = "" if abs(need) < 10 else ("↑" if need > 0 else "↓")
                # `:+.0f`, not `:+d`: an override target comes from
                # stage_params.json as a float, and `:+d` raises on one -- which
                # the except below would swallow into a debug line, leaving the
                # reset phase silent for no visible reason.
                cells.append(f"j{j} {cur:+.0f}→{tgt:+.0f}{mark}")
            parts.append(f"{side[0].upper()}: " + " ".join(cells))
        # Only for the TABLE's medians. An override is the designated single pose
        # the runner is actually driving to, so the "median may be a pose no demo
        # used" caveat does not apply to it.
        note = (
            " (⚠️ 시작 자세가 여러 무리 -- 중앙값은 참고만)"
            if _TARGET_OVERRIDE is None and _STAGE in _MULTIMODAL
            else ""
        )
        logger.info(
            f"{tag} task{_STAGE:02d}{note} | 목표까지 {math.sqrt(sq):.2f} rad | "
            f"가장 큰 차 {worst[1]} {worst[0]:+.0f}° | " + " | ".join(parts)
        )
    except Exception:  # never let a display break a recording
        logger.debug("pose guide failed", exc_info=True)
