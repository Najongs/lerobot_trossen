"""How the arms park (staged, then sleep pose, then driver cleanup), together.

WidowXAIFollower.disconnect and WidowXAILeaderTeleop.disconnect park their arm
on a background thread so that several arms park together. lerobot's
Robot.__del__ and Teleoperator.__del__ call disconnect() on an object dropped
while still connected -- an exception during connect, or a script that never
disconnects -- and starting a thread from a finalizer can deadlock in
Thread.start() (seen at interpreter exit under pytest, Python 3.11). Each class
therefore wraps its __del__ in park_in_place(): disconnect() then only queues
its arm, and leaving the outermost park_in_place() parks every queued arm on
the calling thread, still together (both arms of a bimanual robot or leader).

Ctrl+C while arms park is ignored with a warning (guard_sigint): a second
Ctrl+C used to cut the park short and leave the arms standing in the staged
pose, and it cannot stop a move the controller is already running anyway --
that is the e-stop's job. The guard gives up after PARK_GUARD_S, so a park that
hangs can still be interrupted.

Whatever connect() leaves connected at exit -- a start that failed part way, a
script that never disconnects -- is parked by an atexit handler, every arm
together: it runs after the interpreter has joined the park threads and before
modules are torn down, unlike __del__, which can run after trossen_arm is gone.

An arm passed to park_together() provides ``driver``, ``config.staged_positions``,
``config.joint_names``, ``_begin_park()`` (wait for its connect move, set the
mode) and ``_park_error``; ``_record_command(sent_s, goal_time)`` is optional.
"""

import atexit
import contextlib
import logging
import signal
import threading
import time
import weakref
from collections.abc import Iterator, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# connect() and disconnect() move each arm to the staged pose (and on the way
# out to the sleep pose) over this many seconds.
STAGED_MOVE_S = 2.0
# Slack after a non-blocking move's nominal end before the arm counts as there:
# the host stamps the send after the call returns, and stage_runner.reposition
# uses the same 0.05 s for the same driver.
ARRIVAL_MARGIN_S = 0.05

# How long one park may ignore Ctrl+C: the rest of a connect move, the two legs,
# and slack. Past it Ctrl+C works again even while a park is still running.
PARK_GUARD_S = 3 * (STAGED_MOVE_S + ARRIVAL_MARGIN_S) + 3.0

_state = threading.local()

# Devices that connect() registered for the exit sweep, as (rank, weakref):
# rank 0 a whole robot with a base, 1 an arm pair, 2 one arm. The sweep goes
# outermost first, so a robot's own disconnect() stops its base before its arms.
_tracked: list[tuple[int, "weakref.ref[Any]"]] = []
_tracked_lock = threading.Lock()

_guard_lock = threading.Lock()
_parks_running = 0
_guard_until = 0.0
_guard_installed = False
_previous_sigint: Any = None


def guard_sigint() -> None:
    """Make Ctrl+C wait for running parks. Main thread only; installs once.

    The handler passes Ctrl+C on to the one it replaced whenever no park is
    running, so outside a park nothing changes.
    """
    global _guard_installed, _previous_sigint
    if _guard_installed or threading.current_thread() is not threading.main_thread():
        return
    try:
        current = signal.getsignal(signal.SIGINT)
        # Already ours (state reset by a test, a reload): never chain to itself.
        if current is not _on_sigint:
            _previous_sigint = current
            signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        return
    _guard_installed = True


def _on_sigint(signum: int, frame: Any) -> None:
    if _parks_running > 0 and time.monotonic() < _guard_until:
        logger.warning(
            "Ctrl+C ignored: the arms are parking (a few seconds). "
            "Use the e-stop to stop them."
        )
        return
    previous = _previous_sigint
    if callable(previous):
        previous(signum, frame)
    elif previous != signal.SIG_IGN:
        raise KeyboardInterrupt


@contextlib.contextmanager
def _parking() -> Iterator[None]:
    """Count a running park for the Ctrl+C guard."""
    global _parks_running, _guard_until
    with _guard_lock:
        _parks_running += 1
        _guard_until = max(_guard_until, time.monotonic() + PARK_GUARD_S)
    try:
        yield
    finally:
        with _guard_lock:
            _parks_running -= 1


def park_at_exit(device: Any, rank: int) -> None:
    """Park ``device`` at exit if it is still connected then (see the module doc).

    A device may define ``needs_park_at_exit()`` when ``is_connected`` is the
    wrong test, e.g. MobileAIRobot, whose arms can be up while its cameras are not.
    """
    with _tracked_lock:
        _tracked.append((rank, weakref.ref(device)))


def _park_still_connected() -> None:
    """atexit: disconnect every tracked device still connected, arms together."""
    with _tracked_lock:
        entries = sorted(_tracked, key=lambda entry: entry[0])
        _tracked.clear()
    devices = [device for _, ref in entries if (device := ref()) is not None]
    with park_in_place():
        for device in devices:
            try:
                needs = getattr(device, "needs_park_at_exit", None)
                if not (needs() if callable(needs) else device.is_connected):
                    continue
                logger.warning(f"{device} still connected at exit; parking it")
                device.disconnect()
            except Exception as error:
                logger.error(f"{device} could not be disconnected at exit: {error!r}")


atexit.register(_park_still_connected)


def parking_in_place() -> bool:
    """True inside park_in_place() on this thread."""
    return getattr(_state, "queue", None) is not None


def queue_for_parking(arm: Any) -> None:
    """Park ``arm`` when the outermost park_in_place() on this thread exits."""
    _state.queue.append(arm)


@contextlib.contextmanager
def park_in_place() -> Iterator[None]:
    """Make disconnect() park on the calling thread, all queued arms together."""
    if parking_in_place():
        yield
        return
    _state.queue = []
    try:
        yield
    finally:
        queued, _state.queue = _state.queue, None
        if queued:
            park_together(queued)


def park_together(arms: Sequence[Any]) -> None:
    """Park ``arms`` on the calling thread, all moving at once.

    Each leg -- staged pose, then sleep pose -- is sent to every arm without
    waiting and then waited out once. An arm whose step raises keeps the error
    in ``_park_error``, is logged, and drops out of the remaining steps; the
    others carry on. Ctrl+C is ignored meanwhile (guard_sigint).
    """
    guard_sigint()
    with _parking():
        _park_together(arms)


def _park_together(arms: Sequence[Any]) -> None:
    live = list(arms)

    def drop(arm: Any, error: Exception) -> None:
        arm._park_error = error
        live.remove(arm)
        logger.error(f"{arm} failed to park: {error!r} -- check the arm")

    for arm in list(live):
        try:
            arm._begin_park()
        except Exception as error:
            drop(arm, error)

    legs = (
        lambda arm: list(arm.config.staged_positions),
        lambda arm: [0.0] * len(arm.config.joint_names),
    )
    for goal_of in legs:
        moving = False
        for arm in list(live):
            try:
                arm.driver.set_all_positions(
                    goal_of(arm), goal_time=STAGED_MOVE_S, blocking=False
                )
            except Exception as error:
                drop(arm, error)
                continue
            moving = True
            record = getattr(arm, "_record_command", None)
            if callable(record):
                record(time.perf_counter(), STAGED_MOVE_S)
        if moving:
            time.sleep(STAGED_MOVE_S + ARRIVAL_MARGIN_S)

    for arm in list(live):
        try:
            arm.driver.cleanup()
        except Exception as error:
            drop(arm, error)
            continue
        logger.info(f"{arm} disconnected.")
