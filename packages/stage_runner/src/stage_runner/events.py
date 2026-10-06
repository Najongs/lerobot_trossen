"""The JSONL event stream: the runner's only output channel besides the dataset.

One JSON object per line, flushed per event. Stdlib only, so the file it writes
stays readable by aggregate.py without lerobot or torch.

Success, failure and the 3-way failure classification (inside the preceding
stage / at the transition / inside the following stage) are deliberately NOT
collected here. There is no terminal prompt and no keyboard listener of our own:
those judgements live on video plus the operator's handwriting and join back on
wall_clock_iso. NO metric is ever computed in this module either -- keeping the
log raw and the aggregation separate is what lets a metric defined next month be
recomputed over runs recorded today. The concrete case is the chaining loss
ratio: its denominator is a STANDALONE per-stage success rate, which no chain
trial can contain and which will be acquired in a separate session (see
aggregate.py's docstring). Every log written before that acquisition still
answers the metric afterwards, because nothing here pre-aggregates anything.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from stage_runner.results import TERMINATED_BY_VALUES

logger = logging.getLogger(__name__)

EVENT_SCHEMA_VERSION: int = 1

EVENT_TRIAL_START: str = "trial_start"
EVENT_STAGE_START: str = "stage_start"
EVENT_STAGE_END: str = "stage_end"
EVENT_TRANSITION: str = "transition"
EVENT_TRIAL_END: str = "trial_end"
EVENT_NAMES: tuple[str, ...] = (
    EVENT_TRIAL_START,
    EVENT_STAGE_START,
    EVENT_STAGE_END,
    EVENT_TRANSITION,
    EVENT_TRIAL_END,
)

# Present on every line whatever the event is. Per-event fields are additive on
# top and arrive through **extra.
BASE_FIELD_NAMES: tuple[str, ...] = (
    "run_id",
    "stage_id",
    "wall_clock_iso",
    "frame_idx",
    "terminator",
    "reason",
)

# Keys emit() writes itself. An extra of the same name would silently rewrite
# the join key or the schema version, so a collision is rejected instead.
_RESERVED_FIELD_NAMES: frozenset[str] = frozenset(BASE_FIELD_NAMES) | {
    "event",
    "schema_version",
}


def wall_clock_iso(now: datetime | None = None) -> str:
    """Local time with UTC offset and milliseconds, e.g. 2026-09-08T14:12:03.412+09:00.

    Local rather than UTC because this is the join key to the camera's capture
    times and to a sheet the operator fills in by hand beside the robot. It is
    also why the dataset's own `timestamp` column cannot serve: that is
    frame_index / fps, which lies by exactly the amount the loop runs below fps.
    """
    moment = datetime.now() if now is None else now
    return moment.astimezone().isoformat(timespec="milliseconds")


def _json_scalar(value: Any) -> Any:
    """Last-resort coercion for a value type the schema did not anticipate.

    emit() runs between control loops with the robot connected, so a TypeError
    raised inside json.dumps would take the trial down along with the log.
    numpy scalars (an observation value that skipped its float() cast) become
    plain numbers; anything else degrades to its string form.
    """
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _replace_non_finite(value: Any) -> Any:
    """Recursively turn NaN / Infinity floats into their string form.

    json.dumps defaults to allow_nan=True, which writes the bare tokens NaN and
    Infinity. Those are not valid JSON: Python's own json.loads reads them back,
    so the aggregator never notices, but this file is the durable artifact --
    jq, pandas in strict mode, a spreadsheet import or another team's parser all
    fail on that line or on the whole file. Since the log outliving this package
    is the entire reason for choosing JSONL, "NaN" as a string is strictly
    better than a token nobody else can read.

    Recursive because the offending value arrives nested: stop_action is a dict
    of floats copied out of robot.get_observation (transitions.build_hold_action).
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _replace_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_non_finite(item) for item in value]
    return value


class EventLog:
    """Append-only JSONL writer for one run."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        # "a", not "w": the run directory is created with exist_ok=False so this
        # file is always new, and append means a second open can never truncate
        # a trial that already ran.
        self._file = self.path.open("a", encoding="utf-8")

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the file. Idempotent -- cli.main's finally also calls it."""
        if self._file is None:
            return
        self._file.close()
        self._file = None

    def emit(
        self,
        event: str,
        *,
        stage_id: str | None = None,
        frame_idx: int | None = None,
        terminator: str | None = None,
        reason: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        """Write one event line and return the record that was written.

        Rejects an unknown event name, an unknown terminator and an extra field
        that collides with a base field: a typo in any of the three produces a
        log the aggregator would parse without complaint and summarise wrongly.
        """
        if event not in EVENT_NAMES:
            raise ValueError(
                f"unknown event {event!r}; expected one of {list(EVENT_NAMES)}"
            )
        # TERMINATOR_TYPES ("timeout", "manual") is a subset of
        # TERMINATED_BY_VALUES, so this one check covers the planned terminator
        # on stage_start as well -- and it keeps config.py, which imports
        # draccus and lerobot's RobotConfig, out of this stdlib-only module.
        if terminator is not None and terminator not in TERMINATED_BY_VALUES:
            raise ValueError(
                f"unknown terminator {terminator!r}; "
                f"expected one of {list(TERMINATED_BY_VALUES)} or None"
            )
        collisions = sorted(set(extra) & _RESERVED_FIELD_NAMES)
        if collisions:
            raise ValueError(f"extra fields collide with base fields: {collisions}")

        record: dict[str, Any] = {
            "event": event,
            "schema_version": EVENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "wall_clock_iso": wall_clock_iso(),
            "stage_id": stage_id,
            "frame_idx": frame_idx,
            "terminator": terminator,
            "reason": reason,
        }
        record.update(extra)
        self._write(record)
        return record

    def _write(self, record: dict[str, Any]) -> None:
        if self._file is None:
            raise ValueError(f"EventLog on {self.path} is closed")
        try:
            line = json.dumps(
                record, ensure_ascii=False, default=_json_scalar, allow_nan=False
            )
        except ValueError:
            # allow_nan=False is what makes a NaN detectable at all; without it
            # the token is written silently. Degrade the value rather than drop
            # the line -- a NaN reaching the log is itself the interesting fact,
            # and losing the whole event would hide the stage it belongs to.
            sanitized = _replace_non_finite(record)
            logger.warning(
                f"non-finite value in a {record.get('event')!r} event was written "
                f"as a string: bare NaN/Infinity tokens are not valid JSON and "
                f"this file is read by tools that are not Python"
            )
            line = json.dumps(
                sanitized, ensure_ascii=False, default=_json_scalar, allow_nan=False
            )
        self._file.write(line + "\n")
        # A run writes tens of lines, so flushing every one costs nothing, and a
        # trial killed mid-chain still has every stage that already finished.
        # The data behind those lines is not reproducible.
        self._file.flush()
