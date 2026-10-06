"""The last gate before ``send_action``: no non-finite number reaches the arms.

WHY IT EXISTS. lerobot's own guard is ``ensure_safe_goal_position``, and the clamp
it applies is a ``min``/``max`` pair (``lerobot/robots/utils.py:99-104``). Every
comparison with NaN is False, so ``min(max(nan, lo), hi)`` returns ``nan``: the
clamp does not fire, it does not raise, and the value passes through looking
clamped. ``WidowXAIFollower.send_action`` then does
``abs(goal_pos[j] - present_pos[j]) > max_relative_target``
(widowxai_follower.py:272) -- again False for NaN -- so the 0.1 rad relative
target does not fire either, and the velocity pacing that depends on that same
delta computes nothing. The NaN goal reaches ``set_all_positions``.

The fork already recorded what this costs on the BASE side, where it was found
and fixed in place: "max(-MAX, NaN) returns -MAX -- a NaN command reaches the base
as full-speed reverse" (mobileai.py:54-55, sanitized at :159-179). There is no
equivalent on the ARM side, and the two known producers are real: a corrupted
normalizer stats file and an fp16 overflow, which this repo's README already
documents as a pair the operator has to tell apart (README:359, a9c2a8a).

WHAT IT DOES. One ProcessorStep at the END of ``robot_action_processor``, after
the completion monitor. If every float in the action dict is finite it returns the
transition it was given, unmodified and not copied -- the common case costs one
pass over 17 floats. If any is not, it

1. builds a HOLD action -- every ``.pos`` key at its MEASURED value, the base
   velocities at 0.0, the progress slot at 0.0 -- and returns a transition
   carrying that instead, so ``send_action`` gets the hold and not the NaN;
2. logs ERROR with the offending keys;
3. sets ``events["exit_early"]`` and ``events["stop_recording"]``, which ends the
   stage at the top of record_loop's next iteration and breaks the chain.

IT DOES NOT MUTATE THE ACTION DICT, and that is deliberate. ``record_loop``
assigns ``action_values = act_processed_policy`` BEFORE it calls
``robot_action_processor`` and records THAT object
(lerobot_record.py:395-397, :410-411) -- so the dict this step is handed is the
one the dataset frame is built from. Mutating it would make the recorded frame a
hold action that the model never produced, erasing the only evidence of what went
wrong; returning a new dict keeps the recorded frame truthful (the model's NaN)
while the robot receives the hold. The two are DIFFERENT facts and the dataset
should carry the first.

WHY NOT JUST RAISE. An exception out of a ProcessorStep unwinds through
``record_loop``'s ``@safe_stop_image_writer`` and the whole episode -- every stage
already recorded in it -- is discarded unsaved. A chain that reached stage 7 and
then saw a NaN is data; losing it to report the NaN is the wrong trade. The one
case that does raise is a hold that cannot be built at all (see
:meth:`FiniteActionGateStep._hold_action`), where there is no safe action left to
send and the runner's error path zeroes the base directly.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

from lerobot.processor.pipeline import ProcessorStep

from stage_runner.completion import BASE_VELOCITY_KEYS, PROGRESS_KEY

logger = logging.getLogger(__name__)


def non_finite_keys(action: Mapping[str, Any]) -> tuple[str, ...]:
    """The action keys whose value is not a finite number, in iteration order.

    A value that is not a number at all counts as non-finite: the arms are
    commanded from this dict and a string or None there is not something to pass
    through on the grounds that ``math.isfinite`` could not judge it.

    ``bool`` is a number to Python (``math.isfinite(True)`` is True) and would be
    silently commanded as 1.0 radians. It is treated as non-finite here, because
    nothing in this package ever legitimately puts a bool in an action.
    """
    offenders: list[str] = []
    for key, value in action.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            offenders.append(key)
            continue
        if not math.isfinite(float(value)):
            offenders.append(key)
    return tuple(offenders)


class FiniteActionGateStep(ProcessorStep):
    """Replace a non-finite action with a hold and break the chain.

    Installed ONCE, at the end of ``robot_action_processor``, which every stage
    goes through: the policy stages and the reset ramps both drive
    ``record_loop`` with the same ``context.processors``. The ramp is arithmetic
    on measured values and should never produce a NaN, but "should never" is what
    the arm side of the clamp already assumed.

    ``tripped`` is read by ``cli`` after the trial to turn a trip into exit code 4
    (the chain broke) rather than 1 (a human stopped it): the flags this step sets
    are the same ones Esc sets, and without the distinction a NaN would be filed
    in the bucket an operator reads as "my own ESC".
    """

    def __init__(
        self,
        events: dict[str, bool],
        *,
        base_velocity_keys: Sequence[str] = tuple(BASE_VELOCITY_KEYS),
    ) -> None:
        self.events = events
        self.base_velocity_keys = frozenset(base_velocity_keys)
        # The first trip's offending keys, or (). Sticky: the loop breaks on the
        # next iteration, but the tick that tripped still completes and a second
        # one can arrive before it does.
        self.tripped_keys: tuple[str, ...] = ()
        self.trips: int = 0

    @property
    def tripped(self) -> bool:
        return self.trips > 0

    # ------------------------------------------------------------- ProcessorStep

    def __call__(self, transition):
        from lerobot.processor.core import TransitionKey

        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            # Nothing to gate and nothing this step can assert. The pipeline's own
            # to_output converter raises on a non-dict action one call later,
            # which is a clearer error than anything invented here.
            return transition
        offenders = non_finite_keys(action)
        if not offenders:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION)
        hold = self._hold_action(
            action, observation if isinstance(observation, dict) else {}
        )
        self.trips += 1
        if not self.tripped_keys:
            self.tripped_keys = offenders
        logger.error(
            f"NON-FINITE ACTION at the last gate before send_action: "
            f"{ {key: action[key] for key in offenders} }. The arms were sent a "
            "HOLD (every joint at its measured position, base 0) instead -- "
            "lerobot's own clamp would have passed the value through unchanged, "
            "because every comparison with NaN is False "
            "(robots/utils.py:99-104, widowxai_follower.py:272). The chain is being "
            "stopped. Two known causes, and they need telling apart: a corrupted "
            "normalizer stats file in the checkpoint, or an fp16 overflow "
            "(README:359)."
        )
        # BOTH flags. exit_early alone means "this stage is done, go on", which
        # would walk the robot into the next stage on a model that is emitting
        # NaN; stop_recording is what classify_termination turns into an aborting
        # terminator so the runner breaks out of the stage loop.
        self.events["exit_early"] = True
        self.events["stop_recording"] = True

        patched = dict(transition)
        patched[TransitionKey.ACTION] = hold
        return patched

    def transform_features(self, features):
        # Reads and substitutes values; never adds, removes or renames a key, so
        # the pipeline's feature description is unchanged.
        return features

    def get_config(self) -> dict[str, Any]:
        return {"base_velocity_keys": sorted(self.base_velocity_keys)}

    # ------------------------------------------------------------------ internals

    def _hold_action(
        self, action: Mapping[str, Any], observation: Mapping[str, Any]
    ) -> dict[str, float]:
        """Every key of ``action``, holding position and zeroing the base.

        THE SAME SHAPE as the action it replaces, key for key. A narrower dict is
        a KeyError at widowxai_follower.py:272, which evaluates
        ``goal_pos[j] - present_pos[j]`` for every name in ``config.joint_names``
        -- so dropping the offending key instead of replacing it would stop the
        arms by stopping the whole write, base included
        (``transitions.build_hold_action`` documents the same trap).

        Per key: the base velocities and the progress slot become 0.0; a ``.pos``
        key becomes its MEASURED value; anything else becomes its measured value
        too if the observation has one. A ``.pos`` key with no finite measurement
        falls back to the value already in the action if THAT one is finite --
        which is the case where only the progress slot went NaN.

        RAISES when a key has neither. There is no safe value left to invent: 0.0
        for a gripper opens or closes it on whatever the arm is carrying, and 0.0
        for a joint is a position command to the middle of the range. The runner's
        error path catches it, zeroes the base directly and resolves the episode.
        """
        hold: dict[str, float] = {}
        unresolved: list[str] = []
        for key, commanded in action.items():
            if key in self.base_velocity_keys or key == PROGRESS_KEY:
                hold[key] = 0.0
                continue
            measured = observation.get(key)
            if _is_finite_number(measured):
                hold[key] = float(measured)
                continue
            if _is_finite_number(commanded):
                # The commanded value for THIS key is fine; something else in the
                # dict is what tripped the gate. Repeating it is a no-op for a
                # position-controlled joint.
                hold[key] = float(commanded)
                continue
            unresolved.append(key)
        if unresolved:
            raise ValueError(
                f"the action is non-finite and no hold could be built for "
                f"{unresolved}: the observation carries no finite value for "
                f"{'them' if len(unresolved) > 1 else 'it'} either. "
                "Refusing to invent a position -- 0.0 would command a gripper "
                "onto whatever the arm is holding."
            )
        return hold


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))
