"""Run the left and right arm's connect/configure/disconnect at the same time.

Each WidowX AI arm moves to its staged pose on connect (2 s, blocking) and to staged + sleep on
disconnect (2 s + 2 s, blocking). The bimanual classes used to call the left arm, wait, then the
right arm — 4 s per pair on connect and 8 s per pair on disconnect. Running the two calls in two
threads halves that. The motions themselves (poses, goal_time) are unchanged; only the two arms
now move together. Each arm has its own driver and TCP connection.

Off with ``LEROBOT_PARALLEL_ARMS=0`` (then it is the old left-then-right order).
If one side raises, the other side is still waited for, then the first error is re-raised.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)
_ENV = "LEROBOT_PARALLEL_ARMS"


def enabled() -> bool:
    return os.environ.get(_ENV, "1").strip() != "0"


def run_both(name: str, left_fn, right_fn) -> None:
    """Call ``left_fn()`` and ``right_fn()``; concurrently unless disabled."""
    if not enabled():
        left_fn()
        right_fn()
        return
    errors: list[BaseException] = []

    def _wrap(fn):
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 — re-raised below
            errors.append(e)

    t = threading.Thread(target=_wrap, args=(right_fn,), name=f"{name}-right", daemon=True)
    t.start()
    _wrap(left_fn)
    t.join()
    logger.info(f"[parallel_arms] {name}: 양팔 동시 ({_ENV}=0 으로 끈다)")
    if errors:
        raise errors[0]


# ---------------------------------------------------------------------------
# Followers and leaders together (lerobot-record only)
#
# lerobot-record calls robot.connect() and then teleop.connect(), and in its finally
# robot.disconnect() then teleop.disconnect() (guarded by teleop.is_connected). With the
# pair-level threading above that is still follower pair, then leader pair. Here the
# leader pair's connect/disconnect runs in a thread *during* MobileAIRobot.connect /
# .disconnect, so all four arms move at once. record's own teleop.connect() then becomes a
# no-op (the instance's connect is wrapped to skip when already connected) and its
# teleop.disconnect() is skipped by its own is_connected guard.
#
# Only the teleop made by lerobot_record.make_teleoperator_from_config is paired, so
# stage_runner / other entry points keep their old order. Same switch:
# LEROBOT_PARALLEL_ARMS=0 turns this off too.
# ---------------------------------------------------------------------------
_paired_teleop = None


class _Job:
    def __init__(self, name, fn):
        self.name, self.error = name, None
        self._t = threading.Thread(target=self._run, args=(fn,), name=name, daemon=True)
        self._t.start()

    def _run(self, fn):
        try:
            fn()
        except BaseException as e:  # noqa: BLE001 — re-raised in join()
            self.error = e

    def join(self):
        self._t.join()
        if self.error is not None:
            raise self.error


def start_teleop(action: str):
    """Start the paired teleop's connect/disconnect in a thread; None if nothing to do."""
    tp = _paired_teleop
    if tp is None or not enabled():
        return None
    if action == "connect" and not tp.is_connected:
        fn = tp._parallel_arms_real_connect
    elif action == "disconnect" and tp.is_connected:
        fn = tp.disconnect
    else:
        return None
    logger.info(f"[parallel_arms] 리더암 {action} 을 팔로워와 동시에")
    return _Job(f"leader-{action}", fn)


def apply_record_pairing_patch() -> bool:
    """Wrap lerobot_record.make_teleoperator_from_config to register the teleop. Never raises."""
    if not enabled():
        return False
    try:
        from lerobot.scripts import lerobot_record

        original = lerobot_record.make_teleoperator_from_config
        if getattr(original, "_parallel_arms_wrapped", False):
            return True

        def wrapped(*args, **kwargs):
            global _paired_teleop
            tp = original(*args, **kwargs)
            real_connect = tp.connect

            def connect_once(*a, **k):
                if tp.is_connected:  # already connected alongside the followers
                    return None
                return real_connect(*a, **k)

            tp._parallel_arms_real_connect = real_connect
            tp.connect = connect_once
            _paired_teleop = tp
            return tp

        wrapped._parallel_arms_wrapped = True
        wrapped.__wrapped__ = original
        lerobot_record.make_teleoperator_from_config = wrapped
        return True
    except Exception as exc:  # never break plugin registration
        logger.warning(f"[parallel_arms] record pairing not installed: {exc!r}")
        return False
