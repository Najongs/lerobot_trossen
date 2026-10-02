"""Paired log of commanded vs measured base velocity (off by default, opt in).

Why this exists
---------------
``observation.state`` is 14-dim (arms only) since the base velocity was dropped
from it, so an eval dataset recorded with a 14-dim policy carries the *command*
(``action`` is still 16-dim) but no longer carries what the base actually did.
That pair is exactly what a base latency measurement needs: how long the chassis
takes to reach a commanded velocity, and how far it keeps moving after the
command drops to zero -- the open question behind the base over-rotation work.

Re-enabling ``include_base_in_state`` is not an option: it makes the observation
16-dim and the policy normalizer rejects it. Polling the base from a second
process is not an option either, because the Modbus transaction is already the
control loop's dominant cost. But ``MobileAIRobot.get_observation()`` *already*
reads the measured velocity every iteration and simply discards it when the
observation is 14-dim, so the value is free -- it only needs somewhere to go.

What is logged
--------------
One row per ``send_action()`` call, i.e. exactly one row per control-loop
iteration, with both halves of the pair taken from the same iteration:

- the measured velocity that ``get_observation()`` read at the top of this
  iteration (via the shared ``_latest_base_velocity`` cache), and
- the sanitized command written to the base at the bottom of it -- the same
  value the dataset stores in ``action``, after ``_sanitize_base_command``.

Because both come from one iteration, no timestamp join against the dataset is
needed to pair them.

Rows are buffered in memory and written once at ``disconnect()``. The control
loop never touches the disk: a write inside the loop would perturb the very
timing this log exists to measure.

Caveat
------
``record_loop`` is shared by the recording and the reset phase, so the log spans
both. Reset rows are leader-teleop driven, not policy driven, and the ``phase``
column says which is which -- filter to ``policy`` before judging a policy. The
column marks the boundary between the two, not the boundary between episodes:
consecutive episodes are separated by a ``teleop`` stretch, so a ``policy`` run
is one episode, but a run without a leader arm has no reset rows to separate
them and still needs the dataset to split.

Usage
-----
Off by default. Opt in with a path::

    LEROBOT_BASE_VEL_LOG=~/eval_logs/20260922_task01.csv uv run lerobot-record ...

Columns: ``t_wall`` (unix seconds), ``t_mono`` (perf_counter seconds),
``meas_x_vel``, ``meas_theta_vel``, ``cmd_x_vel``, ``cmd_theta_vel``,
``phase`` (``policy``, ``teleop``, or empty outside a tagged record loop).
"""

import atexit
import logging
import os
import threading
import time
from pathlib import Path

from lerobot_robot_trossen.loop_rate_log import current_phase

logger = logging.getLogger(__name__)

_ENV_VAR = "LEROBOT_BASE_VEL_LOG"
_HEADER = "t_wall,t_mono,meas_x_vel,meas_theta_vel,cmd_x_vel,cmd_theta_vel,phase\n"

# A 20 Hz loop fills this in about 14 hours; the cap only exists so a forgotten
# opt-in cannot grow without bound.
_MAX_ROWS = 1_000_000

_lock = threading.Lock()
_rows: list[tuple[float, float, float, float, float, float, str]] = []
_path: Path | None = None
_overflowed = False
_flushed = False


def _resolve_path() -> Path | None:
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def is_enabled() -> bool:
    return _path is not None


def record_sample(
    meas_x_vel: float,
    meas_theta_vel: float,
    cmd_x_vel: float,
    cmd_theta_vel: float,
) -> None:
    """Buffer one control-loop row. No-op unless the env var is set."""
    if _path is None:
        return
    global _overflowed
    row = (
        time.time(),
        time.perf_counter(),
        meas_x_vel,
        meas_theta_vel,
        cmd_x_vel,
        cmd_theta_vel,
        current_phase() or "",
    )
    with _lock:
        if len(_rows) >= _MAX_ROWS:
            if not _overflowed:
                _overflowed = True
                logger.warning(
                    f"Base velocity log hit {_MAX_ROWS} rows; dropping further samples."
                )
            return
        _rows.append(row)


def flush() -> None:
    """Write the buffered rows to the CSV. Safe to call more than once."""
    global _flushed
    if _path is None:
        return
    with _lock:
        if _flushed or not _rows:
            return
        rows, _flushed = list(_rows), True
    try:
        _path.parent.mkdir(parents=True, exist_ok=True)
        with _path.open("w") as fh:
            fh.write(_HEADER)
            for row in rows:
                fh.write(
                    f"{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},"
                    f"{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},{row[6]}\n"
                )
    except OSError as e:
        # Losing this log must never take the run down with it.
        logger.warning(f"Failed to write base velocity log to {_path}: {e}")
        return
    logger.info(f"Base velocity log written: {_path} ({len(rows)} rows)")


_path = _resolve_path()
if _path is not None:
    # disconnect() is the normal exit, but a crash or Ctrl-C mid-episode should
    # still leave the rows on disk -- those runs are the interesting ones.
    atexit.register(flush)
    logger.info(f"Base velocity log enabled -> {_path}")
