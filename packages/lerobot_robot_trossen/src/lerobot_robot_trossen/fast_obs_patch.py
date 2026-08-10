"""Keep eval-time image preprocessing off the CPU (on by default, opt out).

Why this exists
---------------
``lerobot.policies.utils.prepare_observation_for_inference`` converts camera
frames from uint8 to float32 and reorders HWC -> CHW *on the CPU*, and only then
copies the result to the GPU::

    observation[name] = observation[name].type(torch.float32) / 255   # CPU
    observation[name] = observation[name].permute(2, 0, 1).contiguous()  # CPU copy
    observation[name] = observation[name].to(device)                  # 4x the bytes

For the Mobile AI eval loop (3x 640x480x3) that ships 10.5 MB per frame over PCIe
instead of 2.6 MB, and makes the CPU do an elementwise divide plus a full
``.contiguous()`` copy per camera. Measured on this machine (RTX 5090 Laptop,
real ACT checkpoint, synthetic frames of the exact production shape):

============================  ==========  ====================
segment                       idle p50    p50 under CPU load
============================  ==========  ====================
upstream order (CPU convert)     6.90 ms             54.69 ms
this patch (GPU convert)         0.18 ms              1.59 ms
============================  ==========  ====================

The loaded number is the relevant one -- the eval loop shares the CPU with three
camera threads, base serial I/O and the dataset image writer -- and 54.69 ms
matches the ~54 ms/frame measured on the robot by the control-loop meter in
``mobileai.py`` (the unaccounted ``other=`` term). That single term was the whole
eval-vs-teleop rate asymmetry behind the base over-rotation: base velocity
commands are held until the next ``send_action``, so a loop running at 8.8 Hz
instead of the 21.5 Hz recording rate integrates every rotation ~2.4x too far.

Dtype conversion, permute and device placement all commute here, so reordering
them changes throughput, not results. See the equivalence note below.

Confirmed on hardware (2026-08-10, same ACT checkpoint and 3-camera setup): the
eval loop went from 9.7 Hz to 20.7 Hz and the meter's ``other=`` term from ~56 ms
to 3-4 ms, against a teleop recording baseline of 21.5 Hz -- an over-rotation
multiplier of 1.04 where it had been 2.2. The residual 42 ms of base serial I/O
(``base_read``/``base_write``) is common to both loops and does not contribute to
the multiplier.

Upstream status
---------------
This is the same fix as huggingface/lerobot PR #4339 ("perf(policies): move
prepare_observation_for_inference's image conversion to GPU"), which reports
~35.6 ms -> ~0.3 ms per call and a median control rate of 18.3 Hz -> 70.0 Hz on
SO-101. That PR is still open, and lerobot ``main`` as of 2026-08-10 still does
the conversion CPU-side, so upgrading lerobot does not currently remove the
problem. **Delete this module once #4339 (or an equivalent) ships in the pinned
lerobot release** -- ``apply_fast_observation_patch`` already no-ops in that case.

Usage
-----
On by default since the hardware results above. Opt out for an A/B comparison or
if it is ever suspected of causing trouble::

    LEROBOT_FAST_OBS=0 uv run lerobot-record ...

Combine with ``LEROBOT_LOOP_HZ_LOG=1`` (see ``mobileai.py``) to read the achieved
rate straight out of the log. To check which path a run took::

    grep LEROBOT_FAST_OBS <run log>
"""

import inspect
import logging
import os
import sys

logger = logging.getLogger(__name__)

_ENV_VAR = "LEROBOT_FAST_OBS"
_FUNC_NAME = "prepare_observation_for_inference"

# The patch is applied from this package's import, which lerobot triggers while
# parsing the config (``@parser.wrap()``) -- i.e. *before* ``record()`` calls
# ``init_logging()``. An INFO record emitted at that point is discarded, so the
# confirmation the operator greps for would never reach the log. Announce again
# on the first patched call instead, which happens well after logging is set up.
_announced = False


def _enabled() -> bool:
    """True unless the operator explicitly opts out.

    Enabled by default: the reordering is hardware-verified (140 measurement
    windows across four eval runs, all 20.4-21.3 Hz) and equivalent to one ULP,
    so making it opt-out removes the failure mode where the flag is forgotten and
    the loop silently runs at half rate.
    """
    return os.getenv(_ENV_VAR, "").strip().lower() not in ("0", "false", "no")


def fast_prepare_observation_for_inference(
    observation: dict,
    device,
    task: str | None = None,
    robot_type: str | None = None,
) -> dict:
    """Drop-in replacement that moves each tensor to ``device`` before converting it.

    Mirrors the upstream contract: mutates ``observation`` in place, adds the
    ``task``/``robot_type`` keys, returns the same dict.

    Equivalence with the shipped lerobot 0.4.0 implementation:

    * non-image entries are byte-identical;
    * image entries differ by at most one ULP (measured 5.96e-08 on [0, 1] data)
      because the divide runs on the GPU rather than the CPU -- five orders of
      magnitude below the 1/255 = 3.9e-03 quantisation step of the uint8 source;
    * the ``uint8`` guard means a caller that already hands over float images no
      longer gets divided by 255 a second time. lerobot 0.4.0 divides
      unconditionally; upstream ``main`` added the same guard. The Mobile AI
      cameras always produce uint8, so this path is unreachable here.

    Logs a one-off confirmation on the first call so the operator can verify from
    the run log that the patch is actually in effect (see ``_announced``).
    """
    import torch

    global _announced
    if not _announced:
        _announced = True
        logger.info(
            "GPU-side observation preprocessing is active (converting images on "
            f"the device instead of the CPU). Disable with {_ENV_VAR}=0."
        )

    for name in observation:
        tensor = torch.from_numpy(observation[name]).unsqueeze(0).to(device)
        if "image" in name:
            # (1, H, W, C) -> (1, C, H, W). Permuting while still uint8 means the
            # contiguous copy moves 1 byte/pixel instead of 4.
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
            if tensor.dtype == torch.uint8:
                # .to() out of uint8 always allocates, so div_ cannot alias the
                # caller's data.
                tensor = tensor.to(torch.float32).div_(255)
        observation[name] = tensor

    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""

    return observation


def _converts_on_device(fn) -> bool:
    """True if ``fn`` already transfers before it permutes (i.e. upstream fixed it)."""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    to_device = source.find(".to(device")
    permute = source.find(".permute(")
    return to_device != -1 and permute != -1 and to_device < permute


def apply_fast_observation_patch(force: bool = False) -> bool:
    """Rebind lerobot's observation preparation to the GPU-side version.

    Returns True if the patch is in effect. Never raises: lerobot imports this
    package automatically by name prefix (``lerobot.utils.import_utils``) and
    treats any exception as a failed plugin, which would take the
    ``mobileai_robot`` registration down with it.

    ``prepare_observation_for_inference`` is consumed via ``from ... import``
    (e.g. ``lerobot/utils/control_utils.py:34``), so patching the defining module
    alone would not affect callers that already imported it. Every module holding
    a reference to the original is rebound as well.
    """
    if not (force or _enabled()):
        return False

    try:
        from lerobot.policies import utils as policies_utils

        original = getattr(policies_utils, _FUNC_NAME, None)
        if original is None:
            logger.warning(
                f"lerobot.policies.utils.{_FUNC_NAME} is missing ({_ENV_VAR}); "
                "leaving observation preprocessing untouched."
            )
            return False
        if original is fast_prepare_observation_for_inference:
            return True
        if _converts_on_device(original):
            logger.info(
                f"This lerobot already converts images on the device; skipping the "
                f"{_ENV_VAR} patch (this module can be removed)."
            )
            return False

        setattr(policies_utils, _FUNC_NAME, fast_prepare_observation_for_inference)

        rebound = []
        for module_name, module in list(sys.modules.items()):
            if module is None or module is policies_utils:
                continue
            try:
                if getattr(module, _FUNC_NAME, None) is original:
                    setattr(module, _FUNC_NAME, fast_prepare_observation_for_inference)
                    rebound.append(module_name)
            except Exception:  # modules with exotic __getattr__
                continue

        logger.info(
            f"Patched {_FUNC_NAME} to convert images on the device ({_ENV_VAR}) "
            f"(rebound in: {', '.join(rebound) if rebound else 'lerobot.policies.utils only'})."
        )
        return True
    except Exception:
        logger.exception(
            f"Could not apply the {_ENV_VAR} observation preprocessing patch; "
            "continuing with lerobot's default implementation."
        )
        return False
