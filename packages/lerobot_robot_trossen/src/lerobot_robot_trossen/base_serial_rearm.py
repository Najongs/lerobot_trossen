"""Re-arm the base driver's select() set before each Modbus transaction (opt in).

Why this exists
---------------
Every base transaction in the control loop -- ``update_state()`` in
``get_observation()`` and ``set_cmd_vel()`` in ``send_action()`` -- takes ~20 ms,
together ~41 ms of a ~49 ms eval tick. Half of that is not the device: it is the
driver's own receive timeout. ``trossen_slate`` 0.0.3 keeps the serial fd in a
``fd_set`` member (``SerialDriver::read_set_``) that it fills once and never
refills, while ``select()`` empties the set whenever it times out. The first
timeout happens inside ``init_base()`` (``getVersion()`` never gets a reply), so
from then on every receive calls ``select()`` with an empty set and sleeps the
full 20 ms before reading the bytes that were already waiting.

Setting the fd's bit back before each transaction is what ``FD_SET`` would have
done, and it changes nothing else: the same bytes go out in the same order and the
same values come back, only without the dead wait. Confirmed on the robot on
2026-09-23: ``update_state()`` went from 20.41 ms to 9.97 ms median over 100
calls, ``fail=0``. The ~10 ms left is the real serial round trip, so the loop
saves ~20 ms per tick, which is what lets eval be paced at the recordings' rate.

The binding never releases the GIL, so the dead wait also blocks every other
Python thread for ~41 ms per tick; with the re-arm it is ~20 ms.

Scope and guards
----------------
This writes into another library's C++ object by offset, so it only turns on when
all of these hold, and otherwise leaves the driver untouched and says why:

* ``trossen_slate`` is exactly 0.0.3 (the version whose layout was read from the
  disassembly: ``fd_`` at +0x8, ``read_set_`` at +0x10, bus mutex at +0x90);
* the exported global ``base_driver::driver`` resolves;
* the fd it holds is a ``/dev/tty*`` device, i.e. the offsets really point at the
  serial port and not at something else;
* the running ``record_loop`` is a policy (eval) phase at 21 fps or less, as
  tagged by ``loop_rate_log``. A faster loop over-rotates the other way and runs
  outside the rate the arm velocity pacing was tuned at, and a recording
  (teleop) phase must keep the rate its datasets were taken at, or they stop
  merging with the existing ones. With ``LEROBOT_LOOP_HZ_LOG=0`` there is no
  phase tag, so the re-arm stays off.

Outside a permitted loop the hook puts the bit back the way the driver left it
after ``init_base()``. A set bit would otherwise survive on its own: ``select()``
only empties the set when it times out, and replies keep arriving, so a reset or
teleop phase after an eval phase would silently keep the fast driver.

Why only policy loops at 21 fps or less -- and when to lift that
----------------------------------------------------------------
The limit is not a property of the fix but of the data we have: every dataset
so far was recorded with this driver bug in place, at ~20.4 Hz, and a policy
reproduces those demos best when eval runs at the same period. So eval is paced
at 21 Hz and teleop is left alone, or new episodes would stop matching old ones.
If all the data a policy trains on (KIRO and GIST alike) is re-recorded under
one condition, turn it on for teleop too -- measure the resulting teleop loop
rate first, and pace eval at that rate instead of 21.

Usage
-----
On by default since 2026-09-25 (issue #62): measured on the robot, it removed
the loop-period share of base over-rotation. It still acts only where the guards
above allow, so an eval does nothing different unless it is paced at 21 Hz::

    uv run lerobot-record --dataset.fps=21 ...

``LEROBOT_BASE_SERIAL_REARM=0`` turns it off. The log says when it switches on or off and why
(``grep LEROBOT_BASE_SERIAL_REARM <run log>``), and the ``base_read`` /
``base_write`` sections of the loop rate summary show the saving.
"""

import logging
import os

from lerobot_robot_trossen import loop_rate_log

logger = logging.getLogger(__name__)

ENVIRONMENT_VARIABLE = "LEROBOT_BASE_SERIAL_REARM"

SUPPORTED_TROSSEN_SLATE_VERSION = "0.0.3"
DRIVER_SYMBOL = "_ZN11base_driver6driverE"  # base_driver::driver
FILE_DESCRIPTOR_OFFSET = 0x8
READ_SET_OFFSET = 0x10
READ_SET_BYTES = 128  # sizeof(fd_set) on x86_64 Linux

# Highest loop rate the switch may run at. The fix is paired with 21 Hz eval
# pacing: the teleop recordings run at ~20.4 Hz, and 21 is the nearest integer
# fps that errs towards slight under-rotation rather than over-rotation. Tied to
# today's recordings -- revisit it if the data is re-recorded (see the module doc).
MAXIMUM_LOOP_FPS = 21.0


def _switch_on(value: str) -> bool:
    return value.strip().lower() not in ("", "0", "false", "no")


_SETTING = os.getenv(ENVIRONMENT_VARIABLE)
# Set by hand, as opposed to on by default. Only a hand-set switch announces
# itself at WARNING on every connect; the default stays at INFO so teleop runs,
# where it never acts, do not grow a warning line.
EXPLICIT = _SETTING is not None
REQUESTED = _switch_on(_SETTING) if EXPLICIT else True


# Loop phases the switch may run in. "policy" is the eval rollout it was
# measured against; "reset" is the chain runner's boundary pose ramp
# (stage_runner.reset_policy), which runs through the SAME record_loop at the
# SAME 21 fps and sends a full action -- base velocities included, commanded to
# zero -- on every tick. This is NOT a widening of what the robot may do: the
# ramp commands x.vel = theta.vel = 0.0, so what the switch changes there is only
# the 20 ms the driver spends waiting for a receive it already has. Excluding it
# would leave the boundary ticks running at a different serial cadence from the
# policy ticks either side of them, which is the one thing a per-stage rate
# comparison must not have. Leader-arm teleop ("teleop") stays excluded: a
# RECORDING must keep the rate its data was captured at.
PERMITTED_LOOP_PHASES: tuple[str, ...] = ("policy", "reset")


def loop_refusal_reason(phase: str | None, fps: float | None) -> str | None:
    """Why the switch must stay off in this loop phase, or None if it may run."""
    if phase is None:
        return (
            "no record_loop phase tag (outside a loop, or LEROBOT_LOOP_HZ_LOG=0 "
            "turned the tagging off)"
        )
    if phase not in PERMITTED_LOOP_PHASES:
        return f"{phase} phase (a recording must keep its rate)"
    if fps is None:
        return "loop fps unknown"
    if fps > MAXIMUM_LOOP_FPS:
        return (
            f"--dataset.fps={fps:g} is above {MAXIMUM_LOOP_FPS:g}; pair this switch "
            f"with --dataset.fps={MAXIMUM_LOOP_FPS:g}"
        )
    return None


class BaseSerialRearm:
    """Sets the serial fd's bit in the driver's ``read_set_`` in permitted loops.

    Elsewhere it restores the state found at construction (right after
    ``init_base()``), so the driver behaves exactly as shipped.
    """

    def __init__(self, driver_address: int, file_descriptor: int, device_path: str):
        import ctypes

        self.file_descriptor = file_descriptor
        self.device_path = device_path
        self._read_set = (ctypes.c_ubyte * READ_SET_BYTES).from_address(
            driver_address + READ_SET_OFFSET
        )
        # fd_set is an array of longs, and x86_64 is little-endian, so the fd's bit
        # sits in byte fd // 8 at position fd % 8 -- the same bit FD_SET sets.
        self._byte_index = file_descriptor // 8
        self._bit_mask = 1 << (file_descriptor % 8)
        self.initially_armed = self.is_armed()
        self.rearm_count = 0
        # Last (phase, fps) seen, so the on/off state is logged once per change
        # rather than on every transaction.
        self._last_loop = None
        self._permitted = False

    def is_armed(self) -> bool:
        return bool(self._read_set[self._byte_index] & self._bit_mask)

    def _update_permission(self) -> None:
        loop = (loop_rate_log.current_phase(), loop_rate_log.current_target_fps())
        if loop == self._last_loop:
            return
        was_permitted, self._last_loop = self._permitted, loop
        refusal = loop_refusal_reason(*loop)
        self._permitted = refusal is None
        if self._permitted:
            logger.warning(
                f"{ENVIRONMENT_VARIABLE}: active for this policy loop at "
                f"{loop[1]:g} fps (fd {self.file_descriptor} -> {self.device_path})."
            )
        elif was_permitted:
            logger.info(
                f"{ENVIRONMENT_VARIABLE}: off again ({refusal}); "
                f"{self.rearm_count} re-arms so far this run."
            )
        elif loop[0] == "policy":
            logger.warning(f"{ENVIRONMENT_VARIABLE}: off for this loop: {refusal}.")

    def rearm_if_permitted(self) -> None:
        self._update_permission()
        if self._permitted:
            self._read_set[self._byte_index] |= self._bit_mask
            self.rearm_count += 1
        elif not self.initially_armed:
            self._read_set[self._byte_index] &= ~self._bit_mask & 0xFF


def _refuse(reason: str) -> None:
    logger.warning(f"{ENVIRONMENT_VARIABLE} is on but left off: {reason}.")


def create_base_serial_rearm() -> BaseSerialRearm | None:
    """Build the re-arm hook for the connected base, or None when it must stay off.

    Call after ``TrossenSlate.init_base()``, which is what opens the serial fd.
    Never raises: a failed guard leaves the driver exactly as shipped.
    """
    if not REQUESTED:
        return None
    try:
        import ctypes
        from importlib import metadata

        installed_version = metadata.version("trossen_slate")
        if installed_version != SUPPORTED_TROSSEN_SLATE_VERSION:
            _refuse(
                f"trossen_slate {installed_version} installed, offsets are only "
                f"known for {SUPPORTED_TROSSEN_SLATE_VERSION}"
            )
            return None

        import trossen_slate

        extension_path = trossen_slate.trossen_slate.__file__
        # dlopen of an already loaded library returns the same handle, so this is
        # the driver object the running TrossenSlate uses, not a second copy.
        library = ctypes.CDLL(extension_path)
        driver_address = ctypes.addressof(ctypes.c_char.in_dll(library, DRIVER_SYMBOL))
        file_descriptor = ctypes.c_int.from_address(
            driver_address + FILE_DESCRIPTOR_OFFSET
        ).value
        if file_descriptor < 0 or file_descriptor // 8 >= READ_SET_BYTES:
            _refuse(f"driver fd {file_descriptor} is not an open descriptor")
            return None
        try:
            device_path = os.readlink(f"/proc/self/fd/{file_descriptor}")
        except OSError as error:
            _refuse(f"driver fd {file_descriptor} cannot be resolved ({error})")
            return None
        if not device_path.startswith("/dev/tty"):
            _refuse(
                f"driver fd {file_descriptor} points at {device_path}, not a serial "
                "port; the offsets do not match this build"
            )
            return None

        rearm = BaseSerialRearm(driver_address, file_descriptor, device_path)
        logger.log(
            logging.WARNING if EXPLICIT else logging.INFO,
            f"{ENVIRONMENT_VARIABLE}: ready (fd {file_descriptor} -> {device_path}, "
            f"armed after init: {rearm.initially_armed}); acts only in policy "
            f"loops at --dataset.fps <= {MAXIMUM_LOOP_FPS:g}.",
        )
        return rearm
    except Exception as error:
        _refuse(f"guard failed with {error!r}")
        logger.debug("Base serial re-arm guard failure", exc_info=True)
        return None
