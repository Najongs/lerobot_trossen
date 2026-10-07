"""Loader for ``configs/chain/stage_params.json``: the 11 stages' numbers.

Three things the runner cannot derive and must be TOLD, all three measured in the
training repo (``trossen-ai-simulation/scripts/export_chain_params.py``) and
copied here as one file:

* ``start_pose_rad_arm12`` -- the DESIGNATED pose the boundary reset drives the
  12 arm joints to before stage k starts. Not an average: t02·03·04·05·10 have
  bimodal demonstration start poses, so the per-joint mean or median is a pose no
  demonstration ever used (offline §92.1). One designated pose per stage.
* ``p10_s`` / ``p50_s`` / ``p90_s`` -- the demonstration length spread, which is
  1.3-2.7x wide (§93). ``p10_s`` is the floor the completion monitor refuses to
  declare "done" before; ``p90_s`` times a factor is the stage timeout.
* ``end_poses_rad_arm12`` -- where the demonstrations ENDED, for the 16-D model
  that has no progress output and must be judged by "stopped near a known end
  pose" instead.

STDLIB ONLY, deliberately. This module is imported by the config layer and by
the tests, and keeping lerobot and torch out of it is what lets the JSON be
validated on a machine with no robot stack -- which is where it is generated.

WHY NAMES AND NOT INDICES. ``arm_joint_order`` is part of the file and every
pose is zipped against it, so a file written in a different joint order is read
correctly rather than silently applied to the wrong joints. The real data layout
trap this guards against is already documented: 16-D real recordings are
``[left7, right7, x.vel, theta.vel]`` while the MuJoCo teleop layout puts the
base FIRST, and mixing them "matches the dimension and is quietly wrong"
(CLAUDE.md, 데이터 레이아웃). A pose file is the same class of hazard.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION: int = 1

# The 12 arm joints the reset moves, as ``MobileAIRobot`` names them MINUS the
# ".pos" suffix. The grippers (``left_left_carriage_joint`` /
# ``right_left_carriage_joint``, the 7th value of each arm) are deliberately
# ABSENT: a boundary reset holds the gripper where it is, because the arm may be
# carrying something.
ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
)
ARM_JOINT_COUNT: int = len(ARM_JOINT_NAMES)

STAGE_COUNT: int = 11

_REQUIRED_STAGE_KEYS: tuple[str, ...] = (
    "name",
    "instruction",
    "p10_s",
    "p50_s",
    "p90_s",
    "start_pose_rad_arm12",
    "end_poses_rad_arm12",
    "end_pose_tol_rad",
)


class ChainParamsError(ValueError):
    """The stage parameter file is unusable. Raised before anything is constructed.

    A ValueError and not a PreflightError: this module stays importable without
    lerobot, and ``preflight`` is the layer that converts it into the one-line
    exit 2 an operator reads.
    """


@dataclass(frozen=True)
class StageParams:
    """One stage's numbers. Poses are name -> radians, never a bare list.

    ``start_pose`` and each entry of ``end_poses`` are dicts keyed by the joint
    names in ``ChainParams.arm_joint_order``, so a consumer cannot apply them in
    the wrong order even by accident.
    """

    number: int
    name: str
    instruction: str
    p10_s: float
    p50_s: float
    p90_s: float
    start_pose: Mapping[str, float]
    start_pose_source: str
    end_poses: tuple[Mapping[str, float], ...]
    end_pose_tol_rad: float
    # "move" (1·3·6·10: the base travels, the arm barely moves) or "manip".
    # Optional in the file for compatibility; a file that marks no stage as
    # "move" reproduces the 10/07 false completion (robot PC run 1756: the arm
    # wiggled 0.10 rad, stalled near an end pose, and task03 was declared
    # complete without the base ever turning), so the loader warns about it.
    kind: str = "manip"
    # Demonstration medians of |integral x.vel| (m) and |integral theta.vel|
    # (rad) over the stage, at 21 Hz. The monitor requires a "move" stage to
    # cover `move_base_fraction` of each axis that matters before it may
    # depart or complete. None when the file does not carry them.
    base_fwd_total_m: float | None = None
    base_rot_total_rad: float | None = None
    # The demonstrations' p10 of the same integrals. An axis is REQUIRED of a
    # move stage only when its p10 clears the departure threshold -- i.e. at
    # least 90% of the demonstrations travel on it. task09 drives 0.22 m in the
    # median demonstration but 0 in a tenth of them, so a median rule would make
    # a correct, stationary close-the-fridge uncompletable. Falls back to the
    # median when the file carries no p10.
    base_fwd_p10_m: float | None = None
    base_rot_p10_rad: float | None = None

    def timeout_s(self, factor: float) -> float:
        """The stage's ``control_time_s``: p90 times the configured factor.

        The factor (1.3 by default) is the margin over the slowest tenth of the
        demonstrations. Reaching it is a CHAIN FAILURE, not a retry -- see
        runner.run_trial -- so it is a ceiling on one attempt, not a budget.
        """
        return self.p90_s * float(factor)


@dataclass(frozen=True)
class ChainParams:
    """The whole file, validated."""

    schema_version: int
    fps: int
    source: Mapping[str, Any]
    arm_joint_order: tuple[str, ...]
    stages: Mapping[int, StageParams]
    path: Path

    def stage(self, number: int) -> StageParams:
        try:
            return self.stages[number]
        except KeyError:
            raise ChainParamsError(
                f"{self.path}: no parameters for stage {number}; the file carries "
                f"{sorted(self.stages)}"
            ) from None

    def start_pose_deg_pairs(
        self, number: int
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """``((left j0..j5), (right j0..j5))`` in DEGREES, for ``pose_guide.set_stage``.

        Built by NAME out of :data:`ARM_JOINT_NAMES`, not by slicing the file's
        order, so a file listing right before left still produces the left pair
        first. pose_guide reads ``sent_arms[f"{side}_joint_{j}.pos"]``, which is
        the same naming, so the two cannot disagree about which joint is which.
        """
        pose = self.stage(number).start_pose
        sides = []
        for side in ("left", "right"):
            sides.append(
                tuple(
                    math.degrees(pose[f"{side}_joint_{index}"]) for index in range(6)
                )
            )
        return (sides[0], sides[1])


def _finite(value: Any) -> float | None:
    """float(value) when it is a finite real number, else None.

    bool is rejected: ``json`` parses ``true`` as a bool, ``float(True)`` is 1.0,
    and a typo that puts a bool in a pose would otherwise become a 1-radian
    command.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pose(
    raw: Any, names: Sequence[str], where: str, problems: list[str]
) -> dict[str, float] | None:
    if not isinstance(raw, list) or len(raw) != len(names):
        problems.append(
            f"{where}: expected a list of {len(names)} joint values "
            f"(arm_joint_order), got {type(raw).__name__} of length "
            f"{len(raw) if isinstance(raw, list) else 'n/a'}"
        )
        return None
    pose: dict[str, float] = {}
    for name, value in zip(names, raw):
        number = _finite(value)
        if number is None:
            problems.append(
                f"{where}: joint {name!r} is {value!r}, which is not a finite "
                "number -- a NaN reaching the arms does NOT fail loudly "
                "(the follower's pacing compares with >, and every comparison "
                "with NaN is False)"
            )
            return None
        pose[name] = number
    return pose


def _stage(
    number: int,
    raw: Any,
    names: Sequence[str],
    problems: list[str],
) -> StageParams | None:
    where = f"stages['{number}']"
    if not isinstance(raw, dict):
        problems.append(f"{where}: expected an object, got {type(raw).__name__}")
        return None

    missing = [key for key in _REQUIRED_STAGE_KEYS if key not in raw]
    if missing:
        problems.append(f"{where}: missing key(s) {missing}")
        return None

    before = len(problems)

    name = raw["name"]
    instruction = raw["instruction"]
    for key, value in (("name", name), ("instruction", instruction)):
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{where}.{key}: expected a non-empty string, got {value!r}")

    quantiles: dict[str, float] = {}
    for key in ("p10_s", "p50_s", "p90_s"):
        number_value = _finite(raw[key])
        if number_value is None or number_value <= 0.0:
            problems.append(
                f"{where}.{key}: expected a positive finite number, got {raw[key]!r}"
            )
        else:
            quantiles[key] = number_value
    if len(quantiles) == 3:
        if not quantiles["p10_s"] <= quantiles["p50_s"] <= quantiles["p90_s"]:
            problems.append(
                f"{where}: expected p10_s <= p50_s <= p90_s, got "
                f"{quantiles['p10_s']} / {quantiles['p50_s']} / {quantiles['p90_s']}"
            )
        elif quantiles["p10_s"] >= quantiles["p90_s"]:
            # Equal p10 and p90 means the file carries no spread at all, which
            # makes the completion floor and the timeout the same number: the
            # monitor could then only ever fire in the instant before the
            # timeout. The demonstrations spread 1.3-2.7x (§93), so this is a
            # generation bug, not a tight distribution.
            problems.append(
                f"{where}: p10_s ({quantiles['p10_s']}) must be strictly less "
                f"than p90_s ({quantiles['p90_s']}) -- they are the completion "
                "floor and the timeout, and an equal pair leaves no window"
            )

    start_pose = _pose(
        raw["start_pose_rad_arm12"], names, f"{where}.start_pose_rad_arm12", problems
    )
    start_pose_source = raw.get("start_pose_source", "")
    if not isinstance(start_pose_source, str):
        problems.append(
            f"{where}.start_pose_source: expected a string, got {start_pose_source!r}"
        )
        start_pose_source = ""

    raw_ends = raw["end_poses_rad_arm12"]
    end_poses: list[Mapping[str, float]] = []
    if not isinstance(raw_ends, list) or not raw_ends:
        problems.append(
            f"{where}.end_poses_rad_arm12: expected a non-empty list of poses "
            "(the 16-D model is judged by 'stopped near a known end pose', so an "
            "empty list silently disables its only completion signal)"
        )
    else:
        for index, entry in enumerate(raw_ends):
            pose = _pose(
                entry, names, f"{where}.end_poses_rad_arm12[{index}]", problems
            )
            if pose is not None:
                end_poses.append(pose)

    tolerance = _finite(raw["end_pose_tol_rad"])
    if tolerance is None or tolerance <= 0.0:
        problems.append(
            f"{where}.end_pose_tol_rad: expected a positive finite number, got "
            f"{raw['end_pose_tol_rad']!r}"
        )

    # Optional (10/07): stage kind and the base-travel medians a "move" stage is
    # judged by. `base_*_total_*` may be a number or an object {median, p10, p90}
    # as export_chain_params.py writes it; the median is what the monitor uses.
    kind = raw.get("kind", "manip")
    if kind not in ("move", "manip"):
        problems.append(f"{where}.kind: expected 'move' or 'manip', got {kind!r}")
        kind = "manip"
    totals: dict[str, float | None] = {}
    p10s: dict[str, float | None] = {}
    for key in ("base_fwd_total_m", "base_rot_total_rad"):
        value = raw.get(key)
        p10_value = None
        if isinstance(value, dict):
            p10_value = value.get("p10")
            value = value.get("median")
        if value is None:
            totals[key] = None
            p10s[key] = None
            continue
        number_value = _finite(value)
        if number_value is None or number_value < 0.0:
            problems.append(
                f"{where}.{key}: expected a non-negative finite number (or an "
                f"object with a 'median'), got {raw.get(key)!r}"
            )
            totals[key] = None
            p10s[key] = None
            continue
        totals[key] = number_value
        if p10_value is None:
            p10s[key] = number_value
        else:
            p10_number = _finite(p10_value)
            if p10_number is None or p10_number < 0.0 or p10_number > number_value:
                problems.append(
                    f"{where}.{key}.p10: expected 0 <= p10 <= median, got {p10_value!r}"
                )
                p10s[key] = None
            else:
                p10s[key] = p10_number
    if kind == "move" and totals["base_fwd_total_m"] is None and totals["base_rot_total_rad"] is None:
        problems.append(
            f"{where}: kind is 'move' but neither base_fwd_total_m nor "
            "base_rot_total_rad is given -- the monitor would have no travel to "
            "require, which is the false completion the kind exists to prevent"
        )

    if len(problems) != before:
        return None

    return StageParams(
        number=number,
        name=name,
        instruction=instruction,
        p10_s=quantiles["p10_s"],
        p50_s=quantiles["p50_s"],
        p90_s=quantiles["p90_s"],
        start_pose=start_pose or {},
        start_pose_source=start_pose_source,
        end_poses=tuple(end_poses),
        end_pose_tol_rad=float(tolerance),
        kind=kind,
        base_fwd_total_m=totals["base_fwd_total_m"],
        base_rot_total_rad=totals["base_rot_total_rad"],
        base_fwd_p10_m=p10s["base_fwd_total_m"],
        base_rot_p10_rad=p10s["base_rot_total_rad"],
    )


def parse_chain_params(
    document: Any,
    *,
    path: Path | str = "<memory>",
    expected_fps: int | None = None,
    stage_count: int = STAGE_COUNT,
) -> ChainParams:
    """Validate an already-parsed JSON document. Collects EVERY problem, then raises.

    One pass over the message fixes the whole file, instead of one failed run per
    typo -- the same rule ``preflight.check_stage_definitions`` follows.

    ``expected_fps`` is the run's ``dataset.fps``. The quantiles are SECONDS, so
    they survive an fps change on their own, but the reset ramp's tick budget and
    the monitor's window lengths are both derived from the loop rate, and the
    file records the rate the numbers were measured at. A disagreement is a
    warning-shaped fact the operator must see BEFORE the robot moves, so it is a
    refusal.
    """
    location = Path(path)
    problems: list[str] = []

    if not isinstance(document, dict):
        raise ChainParamsError(
            f"{location}: expected a JSON object at the top level, got "
            f"{type(document).__name__}"
        )

    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        # Hard stop before anything else is read: a later generation may give the
        # same keys different meanings, and this file's values become arm
        # positions.
        raise ChainParamsError(
            f"{location}: schema_version is {version!r}, this stage_runner reads "
            f"{SCHEMA_VERSION}. Regenerate it with the matching "
            "scripts/export_chain_params.py, or run the stage_runner that reads "
            "that generation."
        )

    fps = document.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        problems.append(f"fps: expected a positive integer, got {fps!r}")
    elif expected_fps is not None and fps != expected_fps:
        problems.append(
            f"fps: the file was generated for {fps} Hz but this run uses "
            f"dataset.fps={expected_fps}. The quantiles are in seconds and "
            "survive that, but the reset ramp's tick count and the monitor's "
            "windows are derived from the loop rate -- regenerate the file or "
            "set dataset.fps to match."
        )

    raw_order = document.get("arm_joint_order")
    if not isinstance(raw_order, list) or len(raw_order) != ARM_JOINT_COUNT:
        problems.append(
            f"arm_joint_order: expected {ARM_JOINT_COUNT} joint names, got "
            f"{raw_order!r}"
        )
        raise ChainParamsError(_message(location, problems))
    order = tuple(str(name) for name in raw_order)
    if sorted(order) != sorted(ARM_JOINT_NAMES):
        problems.append(
            f"arm_joint_order: {list(order)} is not the 12 arm joints this robot "
            f"has. Expected the same set as {list(ARM_JOINT_NAMES)} (any order; "
            "every pose is zipped against the file's order, so the ORDER is "
            "free but the NAMES are not). The grippers "
            "(left_left_carriage_joint / right_left_carriage_joint) are not "
            "reset -- a boundary reset holds the gripper, because the arm may be "
            "carrying something."
        )
        raise ChainParamsError(_message(location, problems))

    raw_stages = document.get("stages")
    if not isinstance(raw_stages, dict):
        problems.append(f"stages: expected an object, got {raw_stages!r}")
        raise ChainParamsError(_message(location, problems))

    expected_keys = [str(number) for number in range(1, stage_count + 1)]
    extra = sorted(set(raw_stages) - set(expected_keys))
    absent = [key for key in expected_keys if key not in raw_stages]
    if absent:
        problems.append(
            f"stages: missing stage(s) {absent}. The chain runs 1..{stage_count} "
            "with a reset at every boundary; a missing stage has no start pose, "
            "so there is nothing to reset TO."
        )
    if extra:
        problems.append(
            f"stages: unexpected key(s) {extra}; expected exactly "
            f"'1'..'{stage_count}' as STRINGS (JSON object keys)."
        )

    stages: dict[int, StageParams] = {}
    for key in expected_keys:
        if key not in raw_stages:
            continue
        parsed = _stage(int(key), raw_stages[key], order, problems)
        if parsed is not None:
            stages[int(key)] = parsed

    if problems:
        raise ChainParamsError(_message(location, problems))

    if stages and not any(stage.kind == "move" for stage in stages.values()):
        # Not an error -- a file from before 10/07 has no `kind` -- but said
        # out loud: on such a file every stage is judged as manipulation, and
        # a move stage can then complete without the base ever moving (robot
        # PC run 1756). Regenerate with sim scripts/export_chain_params.py.
        import logging

        logging.getLogger(__name__).warning(
            f"{location}: no stage is marked kind='move' -- the file predates the "
            "base-travel completion rule, so stages 1/3/6/10 can be declared "
            "complete without the base moving. Regenerate stage_params.json."
        )

    return ChainParams(
        schema_version=SCHEMA_VERSION,
        fps=int(fps),
        source=dict(document.get("source") or {}),
        arm_joint_order=order,
        stages=stages,
        path=location,
    )


def _message(location: Path, problems: Sequence[str]) -> str:
    return f"{location}: invalid stage parameters:\n  - " + "\n  - ".join(problems)


def load_chain_params(
    path: Path | str,
    *,
    expected_fps: int | None = None,
    stage_count: int = STAGE_COUNT,
) -> ChainParams:
    """Read and validate the file. The only entry point production code uses."""
    location = Path(path).expanduser()
    try:
        text = location.read_text(encoding="utf-8")
    except OSError as error:
        raise ChainParamsError(
            f"cannot read the stage parameter file {location}: {error}. "
            "`chain.params_path` in the YAML points at it; it is generated by "
            "trossen-ai-simulation/scripts/export_chain_params.py and committed "
            "to configs/chain/."
        ) from error
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ChainParamsError(
            f"{location} is not valid JSON: {error}"
        ) from error
    return parse_chain_params(
        document, path=location, expected_fps=expected_fps, stage_count=stage_count
    )
