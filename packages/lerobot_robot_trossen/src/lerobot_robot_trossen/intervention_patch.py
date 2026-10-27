"""Hand an eval over to the leader arm mid-episode and record the rest (off by default).

Why this exists
---------------
When a policy fails, the robot is left in exactly the state a correction demo
should start from. Stopping the eval and starting a fresh ``lerobot-record``
throws that state away: ``disconnect()`` drives both arms to the staged and
then the sleep pose (widowxai_follower.py), and ``connect()`` back to staged.
Only the base and the objects stay where the policy left them.

With ``LEROBOT_INTERVENTION_REPO_ID=<repo>`` set, the SPACE BAR during a policy
episode stops the policy where it is and lets an operator finish the stage with
the leader arm. That segment is recorded into a separate dataset -- the
intervention dataset -- in the same form as an acquisition episode.

What happens on the space bar
-----------------------------
1. The policy loop exits the way the right arrow ends it; its episode, up to
   that point, is saved to the eval dataset as usual.
2. The base is commanded zero at once. It holds its last velocity until the next
   command, and the leader alignment below takes seconds.
3. The leader is driven, under position control, to the follower's MEASURED
   joints -- grippers included, so a held object stays held -- and handed back
   to gravity compensation (``align_to`` / ``release`` on the leader). The
   follower's first copied command is then not a jump: record_loop sends the
   leader's pose to the follower as it is, and the policy -> leader hand-over is
   where the 9/27 ``Joint 3 velocity limit exceeded`` crashes happened.
4. Base torque is switched off, so the base can be pushed by hand as in
   acquisition -- only when the robot was connected with it on.
5. A reset runs first, exactly as lerobot-record's own reset between episodes:
   ``record_loop`` without the policy and without a dataset, leader -> follower,
   nothing recorded, for ``--dataset.reset_time_s``. The operator takes the
   leader there, and recording starts at the boundary -- right arrow, left arrow
   or the time limit, as when a reset ends in acquisition. ESC in this reset
   stops the run with nothing recorded.
6. ``record_loop`` runs again without the policy: leader -> follower, frames into
   the intervention dataset, task = the eval's ``--dataset.single_task``. The
   recorded action is the leader's joints plus the base's measured velocity
   (MobileAILeaderTeleop.get_action), which is what acquisition records; the base
   ignores that velocity while its torque is off.
7. Keys while recording, as in acquisition: right arrow saves it and the run goes
   on as before (eval episode saved, reset); left arrow goes to a reset and then
   throws the take away and records again (ESC in that reset throws it away and
   stops the run); ESC saves it and stops the run; reaching
   ``--dataset.episode_time_s`` saves it.
8. The base is zeroed and its torque restored. If that fails the run stops: the
   next policy episode would command a base that does not move.

If anything before the hand-over fails (no leader, the leader cannot be aligned,
the dataset cannot be opened), nothing is recorded and the run is stopped rather
than continued: the reset that would follow copies the leader to the follower as
it is, which is the very jump step 3 exists to prevent.

The reset after a take that ends with the right arrow is the eval's own, so a
left arrow there re-records the eval episode, not the take: the take is already
saved.

Loop rate
---------
The intervention loop runs exactly as acquisition did: ``fps=DATASET_FPS`` (30,
``lerobot-record``'s default, the label every acquisition dataset carries) on a
dataset labelled 30. The eval's own ``--dataset.fps`` (21) is not passed on.

The acquisition datasets really ran at about 20.4 Hz, not 30, because of the
base driver's 20 ms receive wait on every transaction (base_serial_rearm.py).
``LEROBOT_BASE_SERIAL_REARM`` removes that wait only in policy loops; this loop
has no policy, so it is a teleop phase and the wait stays, as it did in
acquisition. The rate therefore comes from the same place it came from then, and
the 30 the loop asks for is the same unreachable target.

The label has to be 30 regardless: LeRobot 0.4.4 merges only datasets with equal
fps (``aggregate.validate_all_metadata``), and record_loop refuses a dataset
whose fps differs from the loop's (``lerobot_record.py:303-304``).

The intervention dataset
------------------------
Opened on the first intervention of a run: appended to when it exists (locally,
or on the Hub -- then downloaded first, as ``--resume=true`` does), created
otherwise. It is finalized and pushed together with the eval dataset, with the
same ``--dataset.push_to_hub`` / ``--dataset.private`` / ``--dataset.tags``, by
hooking that dataset's ``finalize`` and ``push_to_hub``.
"""

import copy
import functools
import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_VAR = "LEROBOT_INTERVENTION_REPO_ID"
REPO_ID: str = os.environ.get(_ENV_VAR, "").strip()

# The intervention dataset's label and its loop's fps. See the module
# docstring, "Loop rate".
DATASET_FPS: int = 30

# Leader alignment pacing: the same values as stage_runner's leader_reset stage
# (#72), kept here so this patch does not depend on the stage runner. The move is
# made with the operator's hands near the leader, so it is slower than the
# follower's own moves; 0.5 rad/s is a conservative round number, not measured.
LEADER_ALIGN_AVERAGE_SPEED_RAD_S: float = 0.5
# The leader's own configure() reaches its staged pose over a fixed 2.0 s.
LEADER_ALIGN_MIN_GOAL_TIME_S: float = 2.0

BASE_VELOCITY_KEYS: tuple[str, ...] = ("x.vel", "theta.vel")
POSITION_SUFFIX: str = ".pos"

WRAPPER_MARKER = "_lerobot_trossen_intervention_wrapper"

_lock = threading.Lock()
# The events dict of the policy loop now running, or None outside one: the space
# bar only acts while this is set.
_policy_events: dict | None = None
_requested: bool = False
# perf_counter() when the space bar was accepted. The run log has one-second
# timestamps, so the hand-over timing is logged as elapsed ms from this instead.
_space_at: float | None = None
_listener: Any = None
_listener_started: bool = False

_dataset: Any = None
_saved_this_run: int = 0
_finalized: bool = False


class InterventionError(RuntimeError):
    """The hand-over to the leader arm cannot be done safely."""


# ----- keyboard -----------------------------------------------------------------


def _on_space() -> None:
    global _requested, _space_at
    with _lock:
        events = _policy_events
        if events is None:
            print(
                "Space bar ignored: an intervention starts only during a policy episode."
            )
            return
        if not _requested:
            _space_at = time.perf_counter()
        _requested = True
        events["exit_early"] = True
    print(
        "Space bar pressed. Stopping the policy and handing over to the leader arm..."
    )


def _start_listener() -> None:
    """A second pynput listener next to lerobot's own; it only watches the space bar."""
    global _listener, _listener_started
    if _listener_started:
        return
    _listener_started = True
    try:
        from lerobot.utils.control_utils import is_headless

        if is_headless():
            logger.warning(
                f"{_ENV_VAR}: headless environment, the space bar cannot be read; "
                "interventions are unavailable in this run."
            )
            return
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.space:
                    _on_space()
            except Exception as error:  # a listener thread must not die silently
                print(f"Error handling the space bar: {error}")

        _listener = keyboard.Listener(on_press=on_press)
        _listener.start()
    except Exception:
        logger.exception(
            f"{_ENV_VAR}: could not start the space bar listener; "
            "interventions are unavailable in this run."
        )


# ----- base ---------------------------------------------------------------------


def _zero_base(robot: Any) -> str:
    """``robot.base.set_cmd_vel(0, 0)`` -- disconnect()'s own first step. "" = done.

    Bypasses send_action: the arms need no command (position-controlled, they
    hold) and must not be able to cost the base stop.
    """
    set_cmd_vel = getattr(getattr(robot, "base", None), "set_cmd_vel", None)
    if not callable(set_cmd_vel):
        return ""
    try:
        if set_cmd_vel(0.0, 0.0) is False:
            return "base.set_cmd_vel(0, 0) reported a failed transaction"
    except Exception as error:
        return f"base.set_cmd_vel(0, 0) raised {error!r}"
    return ""


def _base_torque_switch(robot: Any) -> Any | None:
    """``robot.base.enable_motor_torque`` if this run may switch it, else None.

    Only when the robot was CONNECTED with base torque on: restoring "on" to a
    base deliberately left unpowered would power it up behind the operator.
    """
    if not getattr(getattr(robot, "config", None), "enable_base_motor_torque", False):
        return None
    switch = getattr(getattr(robot, "base", None), "enable_motor_torque", None)
    return switch if callable(switch) else None


def _switched(returned: Any) -> tuple[bool, str]:
    """TrossenSlate.enable_motor_torque returns (success, message)."""
    if isinstance(returned, tuple) and returned:
        return bool(returned[0]), str(returned[1]) if len(returned) > 1 else ""
    return returned is not False, ""


def _release_base_torque(robot: Any) -> bool:
    """Zero, then base torque off. True = released; failing to is not fatal."""
    switch = _base_torque_switch(robot)
    if switch is None:
        return False
    problem = _zero_base(robot)
    if problem:
        logger.warning(
            f"intervention: base not zeroed before releasing its torque: {problem}"
        )
    try:
        ok, message = _switched(switch(False))
    except Exception:
        logger.exception("intervention: could not release the base torque; it stays on")
        return False
    if not ok:
        logger.warning(
            f"intervention: base torque release refused ({message}); it stays on"
        )
        return False
    logger.info("intervention: base torque released -- the base can be pushed by hand")
    return True


def _restore_base_torque(robot: Any) -> str:
    """Zero, then base torque on. "" = restored.

    The zero comes first so the base does not re-energise holding whatever
    velocity was last written while it was free.
    """
    switch = _base_torque_switch(robot)
    if switch is None:
        return "this robot has no base torque switch"
    problem = _zero_base(robot)
    if problem:
        return f"base not zeroed before restoring its torque: {problem}"
    try:
        ok, message = _switched(switch(True))
    except Exception as error:
        return f"base.enable_motor_torque(True) raised {error!r}"
    if not ok:
        return f"base.enable_motor_torque(True) failed: {message}"
    logger.info("intervention: base torque restored")
    return ""


# ----- leader -------------------------------------------------------------------


def _is_gripper(key: str) -> bool:
    return "carriage" in key


def follower_targets(robot: Any, teleop: Any) -> dict[str, float]:
    """The follower's MEASURED position for every joint the leader commands.

    The leader's keys and the follower's are the same names -- record_loop sends
    the leader's action to the follower unchanged, which only works because they
    are. A key the follower does not report is refused rather than guessed.
    """
    keys = [
        key
        for key in teleop.action_features
        if key.endswith(POSITION_SUFFIX) and key not in BASE_VELOCITY_KEYS
    ]
    observation = robot.get_observation()
    missing = [key for key in keys if key not in observation]
    if missing:
        raise InterventionError(
            f"the follower reports no measured position for leader key(s) {missing}"
        )
    targets = {key: float(observation[key]) for key in keys}
    non_finite = [key for key, value in targets.items() if not math.isfinite(value)]
    if non_finite:
        raise InterventionError(
            f"the follower's measured position is non-finite for {non_finite}"
        )
    return targets


def align_goal_time(start: dict[str, float], target: dict[str, float]) -> float:
    """From the largest ARM joint delta; grippers are metres and move centimetres."""
    deltas = [
        abs(target[key] - start[key])
        for key in target
        if key in start and not _is_gripper(key)
    ]
    largest = max(deltas, default=0.0)
    return max(LEADER_ALIGN_MIN_GOAL_TIME_S, largest / LEADER_ALIGN_AVERAGE_SPEED_RAD_S)


def align_leader_to_follower(robot: Any, teleop: Any) -> float:
    """Drive the leader to the follower's measured joints, then free it. Blocking.

    Returns the largest remaining arm error (rad) -- logged, not gated: back in
    gravity compensation the leader sags a little, and the operator is about to
    move it anyway.
    """
    align_to = getattr(teleop, "align_to", None)
    if not callable(align_to):
        raise InterventionError(
            f"{type(teleop).__name__} has no align_to(); the leader cannot be moved "
            "to the follower before the hand-over"
        )
    target = follower_targets(robot, teleop)
    before = {key: float(value) for key, value in teleop.get_action().items()}
    goal_time = align_goal_time(before, target)
    print(
        f"Aligning the leader arm to the follower over {goal_time:.1f}s -- "
        "hands off the leader until it stops."
    )
    align_to(target, goal_time)
    after = {key: float(value) for key, value in teleop.get_action().items()}
    return max(
        (
            abs(after[key] - target[key])
            for key in target
            if key in after and not _is_gripper(key)
        ),
        default=0.0,
    )


# ----- intervention dataset -------------------------------------------------------


def recordable_features(features: dict) -> dict:
    """The eval dataset's features as ``LeRobotDataset.create`` takes them.

    Drops the default columns (create adds them back) and the per-video
    ``info`` block, which the eval dataset filled in from its own encoding --
    at the eval's fps, not ours.
    """
    from lerobot.datasets.utils import DEFAULT_FEATURES

    out = {}
    for key, feature in features.items():
        if key in DEFAULT_FEATURES:
            continue
        feature = copy.deepcopy(feature)
        if feature.get("dtype") == "video":
            feature.pop("info", None)
        out[key] = feature
    return out


def _feature_mismatch(expected: dict, existing: dict) -> list[str]:
    problems = []
    for key in sorted(set(expected) | set(existing)):
        if key not in existing:
            problems.append(f"{key}: missing from the intervention dataset")
        elif key not in expected:
            problems.append(f"{key}: not recorded by this run")
        else:
            a, b = expected[key], existing[key]
            if a.get("dtype") != b.get("dtype") or tuple(a.get("shape") or ()) != tuple(
                b.get("shape") or ()
            ):
                problems.append(
                    f"{key}: {b.get('dtype')} {tuple(b.get('shape') or ())} in the dataset, "
                    f"{a.get('dtype')} {tuple(a.get('shape') or ())} in this run"
                )
    return problems


def _on_hub(repo_id: str) -> bool:
    """Whether the Hub has ``repo_id``; raises when the Hub cannot be asked.

    Not "treat it as new": a new dataset starts at episode 0 and its push writes
    over the same chunk and meta files of the one already on the Hub.
    """
    try:
        from huggingface_hub import HfApi

        return bool(HfApi().repo_exists(repo_id, repo_type="dataset"))
    except Exception as error:
        raise InterventionError(
            f"there is no local copy of {repo_id} and the Hub could not be asked "
            f"whether it exists ({type(error).__name__}). Download it first, or run "
            "once with the Hub reachable (HF_HUB_OFFLINE=0) to create it"
        ) from error


def _set_aside_if_empty(root: Path) -> None:
    """Move a local dataset with no episodes out of the way.

    LeRobotDataset.create writes meta/info.json first, so a run that saved no
    take leaves a folder with no tasks or episodes. Opening it fails, and it
    would stop the next run at its first space bar until someone deletes it.
    It holds no recording, but it is renamed rather than deleted.
    """
    info = root / "meta" / "info.json"
    if not info.is_file():
        return
    try:
        total = json.loads(info.read_text())["total_episodes"]
    except (OSError, ValueError, KeyError):
        return
    if total:
        return
    aside = root.with_name(f"{root.name}.empty-{time.strftime('%Y%m%dT%H%M%S')}")
    root.rename(aside)
    logger.warning(f"intervention: {root} had no episodes; moved it to {aside}")


def _image_writer_threads(robot: Any) -> int:
    # lerobot-record's default: 4 threads per camera.
    return 4 * len(getattr(robot, "cameras", None) or {})


def open_intervention_dataset(eval_dataset: Any, robot: Any) -> Any:
    """Append to the intervention dataset if it exists, create it otherwise."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    features = recordable_features(eval_dataset.features)
    # Videos are encoded per episode (batch size 1): with batching, the last
    # episodes would wait for lerobot-record's VideoEncodingManager, which only
    # knows the eval dataset.
    streaming = getattr(eval_dataset, "_streaming_encoder", None) is not None
    options = {
        "batch_encoding_size": 1,
        "vcodec": getattr(eval_dataset, "vcodec", "libsvtav1"),
        "streaming_encoding": streaming,
        "encoder_threads": getattr(eval_dataset, "_encoder_threads", None),
    }
    threads = _image_writer_threads(robot)
    _set_aside_if_empty(HF_LEROBOT_HOME / REPO_ID)
    if (HF_LEROBOT_HOME / REPO_ID).exists() or _on_hub(REPO_ID):
        dataset = LeRobotDataset(REPO_ID, **options)
        if dataset.fps != DATASET_FPS:
            raise InterventionError(
                f"{REPO_ID} is labelled {dataset.fps} fps, not {DATASET_FPS}; it could "
                "not be merged with the acquisition datasets"
            )
        problems = _feature_mismatch(features, recordable_features(dataset.features))
        if problems:
            raise InterventionError(
                f"{REPO_ID} does not match this run's features: " + "; ".join(problems)
            )
        if threads:
            dataset.start_image_writer(num_processes=0, num_threads=threads)
        logger.info(
            f"intervention: appending to {REPO_ID} ({dataset.num_episodes} episodes so far)"
        )
    else:
        dataset = LeRobotDataset.create(
            REPO_ID,
            DATASET_FPS,
            features=features,
            robot_type=getattr(robot, "name", None),
            use_videos=any(
                feature.get("dtype") == "video" for feature in features.values()
            ),
            image_writer_threads=threads,
            **options,
        )
        logger.info(f"intervention: created {REPO_ID} at {DATASET_FPS} fps")
    _hook_eval_dataset(eval_dataset, dataset)
    return dataset


def _hook_eval_dataset(eval_dataset: Any, dataset: Any) -> None:
    """Finalize and push the intervention dataset whenever the eval dataset is."""
    finalize = getattr(eval_dataset, "finalize", None)
    push_to_hub = getattr(eval_dataset, "push_to_hub", None)
    if not callable(finalize) or not callable(push_to_hub):
        logger.warning(
            f"intervention: {type(eval_dataset).__name__} has no finalize/push_to_hub; "
            f"{REPO_ID} is left on disk unfinalized"
        )
        return

    def finalize_both(*args, **kwargs):
        try:
            return finalize(*args, **kwargs)
        finally:
            _finalize_intervention_dataset(dataset)

    def push_both(*args, **kwargs):
        try:
            return push_to_hub(*args, **kwargs)
        finally:
            if _saved_this_run:
                try:
                    dataset.push_to_hub(*args, **kwargs)
                    logger.info(f"intervention: pushed {REPO_ID}")
                except Exception:
                    logger.exception(
                        f"intervention: could not push {REPO_ID}; it is on disk at "
                        f"{dataset.root}"
                    )

    eval_dataset.finalize = finalize_both
    eval_dataset.push_to_hub = push_both


def _finalize_intervention_dataset(dataset: Any) -> None:
    global _finalized
    if _finalized:
        return
    _finalized = True
    try:
        dataset.finalize()
        logger.info(
            f"intervention: {_saved_this_run} episode(s) saved to {REPO_ID} this run "
            f"({dataset.num_episodes} in total)"
        )
    except Exception:
        logger.exception(f"intervention: could not finalize {REPO_ID}")


# ----- the hand-over ------------------------------------------------------------


def reset_time_s(argv: list[str] | None = None) -> float:
    """``--dataset.reset_time_s`` of this run, else lerobot-record's default.

    record_loop is not told the reset time, and the first intervention can come
    before the first reset, so it is read from the command line.
    """
    return _dataset_time_s("reset_time_s", argv)


def episode_time_s(argv: list[str] | None = None) -> float:
    """``--dataset.episode_time_s`` of this run, else lerobot-record's default.

    The take's time limit. Under lerobot-record this is the control_time_s the
    policy episode already ran with; stage_runner calls record_loop with the
    STAGE's timer instead (11.5-32.1 s in the all-stage chain), and a take that
    inherited it would be cut off and saved half-done. stage_runner takes no
    --dataset.* flags, so it gets lerobot-record's default.
    """
    return _dataset_time_s("episode_time_s", argv)


def _dataset_time_s(field: str, argv: list[str] | None) -> float:
    argv = sys.argv[1:] if argv is None else argv
    flag = f"--dataset.{field}"
    for index, arg in enumerate(argv):
        if arg.startswith(flag + "="):
            value = arg[len(flag) + 1 :]
        elif arg == flag and index + 1 < len(argv):
            value = argv[index + 1]
        else:
            continue
        try:
            return float(value)
        except ValueError:
            break
    from lerobot.scripts.lerobot_record import DatasetRecordConfig

    return float(DatasetRecordConfig.__dataclass_fields__[field].default)


def _run_reset(original, reset_call: dict, events: dict) -> None:
    events["exit_early"] = False
    print(
        "Reset, not recording: take the leader arm. Recording starts with -> "
        f"(or by itself after {reset_call['control_time_s']:g}s); ESC stops the run."
    )
    original(**reset_call)


def _intervene(
    original,
    kwargs: dict,
    eval_dataset: Any,
    space_at: float | None = None,
    stopped_at: float | None = None,
) -> None:
    global _dataset, _saved_this_run
    robot = kwargs["robot"]
    teleop = kwargs.get("teleop")
    events = kwargs["events"]

    problem = _zero_base(robot)
    zeroed_at = time.perf_counter()
    if problem:
        logger.error(f"intervention: {problem}")
    # The base keeps its last velocity until the zero command, so these two
    # numbers are how long it kept moving. The policy loop part includes winding
    # down the chunk prefetch thread (chunk_execution_patch).
    if space_at is not None and stopped_at is not None:
        logger.warning(
            "intervention: after the space bar, the policy loop stopped in "
            f"{(stopped_at - space_at) * 1e3:.0f} ms and the base zero command "
            f"went out at {(zeroed_at - space_at) * 1e3:.0f} ms"
        )
    logger.warning(
        f"intervention: eval episode {eval_dataset.num_episodes} of {eval_dataset.repo_id} "
        "stopped by the operator -- count it as a failure"
    )

    try:
        if teleop is None:
            raise InterventionError("this run has no leader arm (--teleop.*)")
        if _dataset is None:
            _dataset = open_intervention_dataset(eval_dataset, robot)
        error = align_leader_to_follower(robot, teleop)
        logger.info(f"intervention: leader aligned (largest arm error {error:.3f} rad)")
    except Exception as error:
        logger.error(
            f"intervention: nothing recorded, stopping the run ({error}). The reset "
            "would copy an unaligned leader to the follower."
        )
        if not isinstance(error, InterventionError):
            logger.exception("intervention: details")
        events["stop_recording"] = True
        return

    call = dict(kwargs)
    # As acquisition ran: see the module docstring, "Loop rate".
    call.update(
        policy=None,
        preprocessor=None,
        postprocessor=None,
        dataset=_dataset,
        fps=DATASET_FPS,
        control_time_s=episode_time_s(),
    )
    # lerobot-record's reset between episodes: no dataset, nothing recorded.
    reset_call = dict(call, dataset=None, control_time_s=reset_time_s())
    # Only after everything that can raise outside the try below: from here on
    # the finally is what turns the torque back on.
    released = _release_base_torque(robot)
    try:
        _run_reset(original, reset_call, events)
        # A left arrow here has no take to throw away; it only starts recording.
        events["rerecord_episode"] = False
        while not events["stop_recording"]:
            events["exit_early"] = False
            print(
                f"Recording the intervention into {REPO_ID}: finish the stage with the "
                "leader arm, then -> to save (<- to redo, ESC to save and stop)."
            )
            original(**call)
            if events.get("rerecord_episode"):
                # As lerobot-record: the take is thrown away after the reset, so
                # an ESC there throws it away too and stops the run.
                _run_reset(original, reset_call, events)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                _dataset.clear_episode_buffer()
                print("Intervention discarded.")
                continue
            break
        frames = int(_dataset.episode_buffer["size"])
        if frames:
            index = _dataset.num_episodes
            _dataset.save_episode()
            _saved_this_run += 1
            logger.warning(
                f"intervention: saved episode {index} of {REPO_ID} ({frames} frames)"
            )
        else:
            _dataset.clear_episode_buffer()
            logger.warning("intervention: no frame recorded, nothing saved")
    finally:
        problem = _restore_base_torque(robot) if released else _zero_base(robot)
        if problem and released:
            logger.error(f"intervention: {problem}; stopping the run")
            events["stop_recording"] = True
        elif problem:
            logger.error(f"intervention: {problem}")


# ----- record_loop wiring -------------------------------------------------------


def _wrap_record_loop(original):
    """Watch the space bar during policy episodes and hand over when it is pressed.

    record() calls record_loop with keywords at both call sites; anything else
    (no policy, no dataset, positional arguments) passes straight through.
    """

    @functools.wraps(original)
    def record_loop(*args, **kwargs):
        global _policy_events, _requested, _space_at
        events = kwargs.get("events")
        eval_dataset = kwargs.get("dataset")
        if (
            args
            or kwargs.get("policy") is None
            or eval_dataset is None
            or events is None
        ):
            return original(*args, **kwargs)
        _start_listener()
        with _lock:
            _requested = False
            _space_at = None
            _policy_events = events
        try:
            result = original(*args, **kwargs)
        finally:
            stopped_at = time.perf_counter()
            with _lock:
                requested = _requested
                space_at = _space_at
                _requested = False
                _space_at = None
                _policy_events = None
        if requested:
            _intervene(original, kwargs, eval_dataset, space_at, stopped_at)
        return result

    setattr(record_loop, WRAPPER_MARKER, True)
    return record_loop


def is_installed(function) -> bool:
    seen = set()
    while function is not None and id(function) not in seen:
        seen.add(id(function))
        if getattr(function, WRAPPER_MARKER, False):
            return True
        function = getattr(function, "__wrapped__", None)
    return False


def apply_intervention_patch() -> bool:
    """Install the wrapper if ``LEROBOT_INTERVENTION_REPO_ID`` is set; True if installed.

    Installed last, outside the other record_loop wrappers, so the intervention
    loop it starts runs through them too. Never raises: lerobot imports this
    package automatically and treats any exception as a failed plugin.
    """
    if not REPO_ID:
        return False
    try:
        from lerobot.scripts import lerobot_record

        original = getattr(lerobot_record, "record_loop", None)
        if original is None:
            logger.warning(
                f"lerobot.scripts.lerobot_record.record_loop is missing; {_ENV_VAR} "
                "is not installed."
            )
            return False
        if is_installed(original):
            return True
        wrapped = _wrap_record_loop(original)
        lerobot_record.record_loop = wrapped
        for module in list(sys.modules.values()):
            if module is None or module is lerobot_record:
                continue
            try:
                if getattr(module, "record_loop", None) is original:
                    module.record_loop = wrapped
            except Exception:  # modules with exotic __getattr__
                continue
        logger.warning(
            f"{_ENV_VAR}={REPO_ID}: space bar during a policy episode hands over to "
            f"the leader arm and records into that dataset at {DATASET_FPS} fps, as "
            "acquisition does."
        )
        return True
    except Exception:
        logger.exception(
            f"Could not install {_ENV_VAR}; running without interventions."
        )
        return False
