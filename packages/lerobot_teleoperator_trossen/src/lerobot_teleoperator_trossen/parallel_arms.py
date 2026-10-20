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
