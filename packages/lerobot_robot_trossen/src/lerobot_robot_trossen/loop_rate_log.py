"""Control-loop rate meter, and the fps warning it replaces (on by default, opt out).

Why this exists
---------------
``send_action()`` runs exactly once per record/eval loop iteration, and the base
velocity command set there is held until the *next* ``send_action()``, so the
wall-clock interval between consecutive calls is exactly the integration window
that turns a base velocity command into rotation. A loop running at half the rate
of the run that produced the training data therefore turns every base rotation
roughly twice as far -- the "base over-rotation" symptom.

The number that answers this is **the achieved rate of the recording run divided
by the achieved rate of the eval run**, which is why both runs need it in their
log. Upstream lerobot instead warns whenever a single frame misses the target fps
budget::

    lerobot/scripts/lerobot_record.py:421-424
    "Record loop is running slower (20.4 Hz) than the target FPS (30 Hz). ..."

That compares the achieved rate against the *configured* target, which on the
Mobile AI is the wrong axis: the teleop recording ceiling is ~21.5 Hz against a
target of 30, so the warning fires on every frame of a perfectly healthy run --
about 20 lines a second, burying the warnings that are designed to be rare and
read (base emergency stop, base command NaN guard, velocity pacing ``FIRED``).

So this module drops that per-frame warning and emits one summary line per 30
frames instead, carrying the achieved rate that the comparison actually needs.

Coupling
--------
The filter is deliberately tied to the meter: ``LEROBOT_LOOP_HZ_LOG=0`` turns the
summary off *and* restores upstream's warning. There is no configuration in which
the loop rate goes unreported -- a silent run with a silently halved rate is the
failure this module exists to prevent.

Usage
-----
On by default. Opt out with::

    LEROBOT_LOOP_HZ_LOG=0 uv run lerobot-record ...

Reading the log -- take the last summary of each run and divide::

    grep "Control loop rate" record_run.log | tail -1   # mean=21.5 Hz
    grep "Control loop rate" eval_run.log   | tail -1   # mean=20.9 Hz
                                                        # -> 1.03x over-rotation

Only compare ``phase=policy`` lines against the recording run: an eval driven
with a leader arm (README section 3-3) also runs the reset phase through
``record_loop``, and those frames carry no policy inference, so they are much
faster and would flatter the comparison.
"""

import functools
import inspect
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

_ENV_VAR = "LEROBOT_LOOP_HZ_LOG"

# Substring identifying the upstream per-frame warning this module replaces. Kept
# short so a reworded tail upstream does not silently un-filter it.
_UPSTREAM_WARNING_MARKER = "Record loop is running slower"

LOOP_HZ_WINDOW = 30  # frames per summary line (~1 s at 30 fps)
_LOOP_HZ_RESET_GAP_S = 1.0  # gaps longer than this (episode reset) are not counted

# A window is escalated to WARNING when it drops below this fraction of the best
# window seen so far in the same phase of the same run. The reference is the run's
# own rate rather than the target fps on purpose: below-target is the normal state
# on this hardware, so a target-based threshold would warn continuously and put us
# back where we started.
_DEGRADED_FRACTION = 0.8
_REFERENCE_MIN_WINDOWS = 3  # windows to observe before a reference is trusted

# "sec" accumulates per-loop-section wall time (seconds) over the window so the
# summary can pinpoint *where* a slow loop spends its time (arms read/write vs
# base I/O vs camera reads) in a single run, instead of black-box camera-drop trials.
_meter = {"prev_t": None, "count": 0, "sum_dt": 0.0, "max_dt": 0.0, "sec": {}}

# Set by the record_loop wrapper below; None when the loop is not running (or when
# the wrapper could not be installed).
_target_fps: float | None = None
_phase: str | None = None

# Best window mean seen so far in the current phase, and how many windows fed it.
_reference_hz: float | None = None
_reference_windows = 0


def enabled() -> bool:
    """True unless the operator explicitly opts out.

    Mirrors ``LEROBOT_FAST_OBS``: on by default, since a run whose loop rate is
    never reported is exactly the case this module is here to catch.
    """
    return os.getenv(_ENV_VAR, "").strip().lower() not in ("0", "false", "no")


LOOP_HZ_LOG_ENABLED = enabled()


def _reset_window() -> None:
    _meter["count"] = 0
    _meter["sum_dt"] = 0.0
    _meter["max_dt"] = 0.0
    _meter["sec"] = {}


def _reset_phase(phase: str | None, target_fps: float | None) -> None:
    """Start a new phase: drop the partial window and the rate reference with it."""
    global _phase, _target_fps, _reference_hz, _reference_windows
    _phase = phase
    _target_fps = target_fps
    _reference_hz = None
    _reference_windows = 0
    _meter["prev_t"] = None
    _reset_window()


def add_loop_section(name: str, seconds: float) -> None:
    """Accumulate wall time for a named loop section (see ``record_loop_tick``)."""
    if not LOOP_HZ_LOG_ENABLED:
        return
    sec = _meter["sec"]
    sec[name] = sec.get(name, 0.0) + seconds


def record_loop_tick() -> None:
    """Measure and periodically log the real control-loop rate.

    Called once per iteration from ``send_action``. Emits a summary every
    ``LOOP_HZ_WINDOW`` frames with the mean and min instantaneous Hz over the
    window, plus the mean time spent in each instrumented section. The section
    breakdown (arms/base/cameras) says which I/O is the bottleneck; ``other`` is
    loop time outside any instrumented section.
    """
    if not LOOP_HZ_LOG_ENABLED:
        return
    global _reference_hz, _reference_windows
    m = _meter
    now = time.perf_counter()
    prev = m["prev_t"]
    m["prev_t"] = now
    if prev is None:
        return
    dt = now - prev
    if dt > _LOOP_HZ_RESET_GAP_S:
        # Episode boundary / reset pause: drop the partial window so a long idle
        # gap does not masquerade as a slow loop.
        _reset_window()
        return
    m["count"] += 1
    m["sum_dt"] += dt
    m["max_dt"] = max(m["max_dt"], dt)
    if m["count"] < LOOP_HZ_WINDOW:
        return

    mean_hz = m["count"] / m["sum_dt"]
    min_hz = 1.0 / m["max_dt"]

    # Escalate only on degradation within this run (see _DEGRADED_FRACTION).
    degraded_from = None
    if (
        _reference_hz is not None
        and _reference_windows >= _REFERENCE_MIN_WINDOWS
        and mean_hz < _DEGRADED_FRACTION * _reference_hz
    ):
        degraded_from = _reference_hz
    _reference_windows += 1
    if _reference_hz is None or mean_hz > _reference_hz:
        _reference_hz = mean_hz

    context = []
    if _phase is not None:
        context.append(f"phase={_phase}")
    if _target_fps:
        context.append(f"target={_target_fps:g} Hz")
    context_str = f" ({', '.join(context)})" if context else ""

    sections = "  ".join(
        f"{name}={m['sec'][name] / m['count'] * 1e3:.0f}ms" for name in sorted(m["sec"])
    )
    # "other" = loop time not inside any instrumented section (policy
    # select_action/preprocessing, dataset.add_frame, processors, busy_wait).
    other_s = m["sum_dt"] - sum(m["sec"].values())
    other = f"other={other_s / m['count'] * 1e3:.0f}ms"

    message = (
        f"Control loop rate over last {m['count']} frames{context_str}: "
        f"mean={mean_hz:.1f} Hz, min={min_hz:.1f} Hz "
        f"(divide the recording run's mean by this one for the base over-rotation multiplier)"
        + (f" | per-frame: {sections}  {other}" if sections else "")
    )
    if degraded_from is not None:
        logger.warning(
            f"{message} -- slowed to {mean_hz / degraded_from:.0%} of the "
            f"{degraded_from:.1f} Hz reached earlier in this run"
        )
    else:
        logger.info(message)
    _reset_window()


class _UpstreamFpsWarningFilter(logging.Filter):
    """Drop upstream's per-frame "running slower than target FPS" warning.

    Attached to the root logger because ``lerobot_record`` emits it through the
    module-level ``logging.warning()``. Filters live on the logger, not on its
    handlers, so this survives ``init_logging()`` clearing the handlers
    (``lerobot/utils/utils.py:159-163``).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        try:
            message = record.getMessage()
        except Exception:  # a record whose args do not format
            return True
        return _UPSTREAM_WARNING_MARKER not in message


def _wrap_record_loop(original):
    """Tag each ``record_loop`` call with its phase and target fps.

    ``record()`` drives both the episode and the reset phase through the same
    ``record_loop`` (``lerobot/scripts/lerobot_record.py:534`` and ``:563``), and
    the reset call passes no ``policy``. Without this tag an eval run with a
    leader arm mixes fast reset windows into the summary and there is no way to
    tell them apart afterwards.
    """
    signature = inspect.signature(original)

    def _call_arguments(args, kwargs) -> dict:
        """Resolve the call's named arguments, decorated wrappers included.

        ``record_loop`` is decorated with ``@safe_stop_image_writer``, whose
        wrapper is a plain ``(*args, **kwargs)`` function with no
        ``functools.wraps`` (``lerobot/datasets/image_writer.py:25-37``), so the
        signature we can see collapses every argument into a ``kwargs`` entry.
        Flatten that entry back out; both ``record()`` call sites pass keywords.
        """
        try:
            arguments = dict(signature.bind(*args, **kwargs).arguments)
        except TypeError:
            arguments = {}
        arguments.update(arguments.pop("kwargs", None) or {})
        for key in ("fps", "policy"):
            if key not in arguments and key in kwargs:
                arguments[key] = kwargs[key]
        return arguments

    @functools.wraps(original)
    def record_loop(*args, **kwargs):
        try:
            arguments = _call_arguments(args, kwargs)
            fps = arguments.get("fps")
            phase = "policy" if arguments.get("policy") is not None else "teleop"
            _reset_phase(phase, float(fps) if fps else None)
        except Exception:  # never let instrumentation break a recording
            logger.exception(
                "Could not tag the record loop phase; continuing untagged."
            )
        try:
            return original(*args, **kwargs)
        finally:
            _reset_phase(None, None)

    return record_loop


def apply_loop_rate_logging_patch() -> bool:
    """Install the summary's context hooks and silence the warning it replaces.

    Returns True if the upstream warning is filtered. Never raises: lerobot
    imports this package automatically by name prefix
    (``lerobot.utils.import_utils``) and treats any exception as a failed plugin,
    which would take the ``mobileai_robot`` registration down with it.
    """
    if not LOOP_HZ_LOG_ENABLED:
        # Opted out: leave upstream's warning in place so the run is not silent
        # about its loop rate.
        return False

    try:
        root = logging.getLogger()
        if not any(isinstance(f, _UpstreamFpsWarningFilter) for f in root.filters):
            root.addFilter(_UpstreamFpsWarningFilter())

        from lerobot.scripts import lerobot_record

        original = getattr(lerobot_record, "record_loop", None)
        if original is None:
            logger.warning(
                "lerobot.scripts.lerobot_record.record_loop is missing; the loop "
                "rate summary will carry no phase or target fps."
            )
        elif not getattr(original, "__wrapped__", None):
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
        return True
    except Exception:
        logger.exception(
            f"Could not install the {_ENV_VAR} loop rate logging patch; "
            "continuing with lerobot's per-frame fps warning."
        )
        return False
