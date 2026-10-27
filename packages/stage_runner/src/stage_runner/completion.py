"""Stage completion classifier: decides from the cameras that a policy stage is done.

A ``terminator: {type: classifier}`` stage still runs under its timer; the
classifier can only end it EARLIER. While the policy acts, a watcher thread
scores the latest camera frames a few times a second with a checkpoint trained
per stage (lerobot's in-tree ``reward_classifier``: frozen ResNet-10 plus a
small head). Once the score has stayed at or above the threshold for
``hold_s`` seconds -- and not before ``min_s`` into the stage -- the watcher
sets ``events["exit_early"]``, which record_loop already breaks on, and the
stage ends with terminated_by="classifier". In ``mode: shadow`` it only writes
down when it would have fired, and the timer ends the stage as before.

Where the frames come from
--------------------------
The policy PREPROCESSOR, not the robot. record_loop calls predict_action every
tick while a policy runs, and predict_action hands the preprocessor the
observation already as [1, 3, H, W] float tensors in [0, 1] on the policy's
device (control_utils.prepare_observation_for_inference). Tapping it there
means:

* no second reader on the cameras, and no robot proxy;
* frames ONLY while the policy acts. lerobot_robot_trossen's intervention patch
  runs its reset and its takes through record_loop with no policy, so the
  preprocessor is never called there and the watcher cannot end a take that the
  operator is recording on the leader arm.

Grippers
--------
A ``grip`` block adds what the cameras cannot see well, read from
observation.state in the same tapped observation (GripTracker, every tick):

* ``type: classifier`` + grip -- a gate. The classifier may end the stage only
  while every named gripper holds something of the configured width (stage 2:
  both tubes in hand). On t06 the classifier fired at 21.1 s while the left
  gripper was still opening and closing on nothing; the gate holds it back.
* ``type: grasp`` -- the stage ends ``wait_s`` after the latest grasp: a
  gripper that opened and then closed and stayed closed. For stage 6 the
  beaker leaves the gripper reading about 0, the same as closing on nothing,
  so the closing is all there is to see, and the shaking after it takes a fixed
  time. A ``model_path`` here is scored for the record and never fires.

Image preprocessing lives here and nowhere else. The training script that fits
a checkpoint imports prepare_image from this module, so the frames a model was
trained on and the frames it scores at runtime go through the same resize and
the same normalization.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from stage_runner.config import (
    COMPLETION_MODE_FIRE,
    TERMINATOR_CLASSIFIER,
    TERMINATOR_GRASP,
    WATCHED_TERMINATORS,
)

logger = logging.getLogger(__name__)

# The in-tree reward classifier's SpatialLearnedEmbeddings is built for a 4 x 4
# feature map (modeling_classifier.py, height=4 / width=4), and ResNet-10 divides
# the input by 32, so 128 x 128 is the only size its head accepts.
IMAGE_SIZE: tuple[int, int] = (128, 128)
# helper2424/resnet10 is an ImageNet-pretrained backbone.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Written beside the classifier's config.json / model.safetensors by the
# training script. Says which cameras in which order, and the decision values
# the offline evaluation chose; the YAML may override the three values.
COMPLETION_META_FILENAME: str = "completion.json"
COMPLETION_META_FORMAT: int = 1


# How often the watcher scores. The policy ticks at ~21 Hz; a stage end is a
# state that lasts, and hold_s already asks for it to last, so 5 Hz loses
# nothing and keeps the classifier's share of the GPU small next to pi0.
DEFAULT_PERIOD_S: float = 0.2

# Where the joint values sit in a policy observation, in dataset feature order.
OBS_STATE_KEY: str = "observation.state"


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] float in [0, 1] -> [B, 3, 128, 128], ImageNet-normalized.

    The input is what lerobot's prepare_observation_for_inference hands a
    policy preprocessor, which is where the runtime taps it.
    """
    resized = F.interpolate(
        image, size=IMAGE_SIZE, mode="bilinear", align_corners=False, antialias=True
    )
    mean = resized.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = resized.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (resized - mean) / std


@dataclass(frozen=True)
class CompletionMeta:
    """What completion.json says about one checkpoint."""

    camera_keys: tuple[str, ...]
    threshold: float
    hold_s: float
    min_s: float
    task: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str, where: str) -> CompletionMeta:
        raw = json.loads(text)
        if raw.get("format") != COMPLETION_META_FORMAT:
            raise ValueError(
                f"{where}: `format: {raw.get('format')}` -- this stage_runner reads "
                f"format {COMPLETION_META_FORMAT}"
            )
        keys = tuple(raw.get("camera_keys") or ())
        if not keys:
            raise ValueError(f"{where}: `camera_keys` is empty")
        return cls(
            camera_keys=keys,
            threshold=float(raw["threshold"]),
            hold_s=float(raw["hold_s"]),
            min_s=float(raw["min_s"]),
            task=str(raw.get("task", "")),
            raw=raw,
        )


def read_meta(model_path: str) -> CompletionMeta:
    """completion.json from a local checkpoint directory or a Hub repo id."""
    local = Path(model_path) / COMPLETION_META_FILENAME
    if local.is_file():
        return CompletionMeta.from_json(local.read_text(), str(local))
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(model_path, COMPLETION_META_FILENAME)
    return CompletionMeta.from_json(
        Path(downloaded).read_text(), f"{model_path}/{COMPLETION_META_FILENAME}"
    )


class CompletionModel:
    """One loaded classifier checkpoint and the cameras it reads, in its order."""

    def __init__(
        self,
        path: str,
        meta: CompletionMeta,
        classifier: Any,
        device: torch.device,
    ) -> None:
        self.path = path
        self.meta = meta
        self.classifier = classifier
        self.device = device

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> CompletionModel:
        """Load the checkpoint at a local directory or a Hub repo id.

        The classifier's __init__ instantiates its backbone with
        ``AutoModel.from_pretrained("helper2424/resnet10", trust_remote_code=True)``
        before the checkpoint's own weights overwrite it, so that repo has to be
        in the HF cache when the robot PC runs with HF_HUB_OFFLINE=1.
        """
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        meta = read_meta(path)
        classifier = Classifier.from_pretrained(path)
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
        classifier.to(torch_device).eval()
        return cls(path, meta, classifier, torch_device)

    def prepare(self, observation: Mapping[str, Any]) -> list[torch.Tensor]:
        """The model's input from a preprocessor observation; KeyError if a camera is missing."""
        return [
            prepare_image(observation[key].to(self.device, non_blocking=True))
            for key in self.meta.camera_keys
        ]

    def score(self, prepared: list[torch.Tensor]) -> float:
        with torch.inference_mode():
            probabilities = self.classifier.predict(prepared).probabilities
            return float(probabilities.reshape(-1)[0])


class GripTracker:
    """Gripper widths from observation.state, updated on the control thread.

    ``update`` runs every tick (it is a handful of float compares), so a 0.5 s
    settle is judged on ~10 values rather than the classifier's 5 Hz.
    """

    def __init__(self, grip: Any, indices: list[int]) -> None:
        self.grip = grip
        self.names: list[str] = list(grip.state_names)
        self.indices = indices
        n = len(indices)
        self._in_band_since: list[float | None] = [None] * n
        self._opened = [False] * n
        self._below_since: list[float | None] = [None] * n
        self._grasped_at: list[float | None] = [None] * n
        self.events: list[list[Any]] = []
        self.samples: list[list[float]] = []
        self._last_sample_t: float | None = None

    @classmethod
    def for_stage(cls, grip: Any, state_names: list[str]) -> GripTracker:
        """Resolve the named grippers against the dataset's observation.state names."""
        missing = [name for name in grip.state_names if name not in state_names]
        if missing:
            raise ValueError(
                f"grip.state_names {missing} are not in observation.state "
                f"({state_names})"
            )
        return cls(grip, [state_names.index(name) for name in grip.state_names])

    def update(self, t: float, state: Any) -> None:
        """One tick: ``state`` is the observation.state row (any indexable)."""
        grip = self.grip
        values = [float(state[i]) for i in self.indices]
        for k, value in enumerate(values):
            if (
                grip.hold_min_m is not None
                and grip.hold_max_m is not None
                and grip.hold_min_m <= value <= grip.hold_max_m
            ):
                if self._in_band_since[k] is None:
                    self._in_band_since[k] = t
            else:
                self._in_band_since[k] = None
            if value > grip.open_above_m:
                if self._grasped_at[k] is not None:
                    self.events.append([round(t, 3), self.names[k], "reopened"])
                self._opened[k] = True
                self._below_since[k] = None
                self._grasped_at[k] = None
            elif value < grip.close_below_m:
                if self._opened[k]:
                    if self._below_since[k] is None:
                        self._below_since[k] = t
                    if t - self._below_since[k] >= grip.settle_s:
                        self._grasped_at[k] = self._below_since[k]
                        self._opened[k] = False
                        self.events.append(
                            [round(self._below_since[k], 3), self.names[k], "grasped"]
                        )
            else:
                self._below_since[k] = None
        if self._last_sample_t is None or t - self._last_sample_t >= DEFAULT_PERIOD_S:
            self._last_sample_t = t
            self.samples.append([round(t, 3)] + [round(v, 5) for v in values])

    def holding(self, t: float) -> bool:
        """Every named gripper inside the hold band for settle_s, as of time t."""
        return all(
            since is not None and t - since >= self.grip.settle_s
            for since in self._in_band_since
        )

    def latest_grasp_s(self) -> float | None:
        grasps = [g for g in self._grasped_at if g is not None]
        return max(grasps) if grasps else None


@dataclass(frozen=True)
class WatcherSettings:
    # Classifier decision values; None for a grasp stage that scores no model.
    threshold: float | None
    hold_s: float | None
    min_s: float | None
    mode: str
    period_s: float = DEFAULT_PERIOD_S
    # Which rule may end the stage: the classifier (TERMINATOR_CLASSIFIER) or
    # the grasp timer (TERMINATOR_GRASP), whose classifier is record-only.
    rule: str = TERMINATOR_CLASSIFIER
    wait_s: float | None = None


def resolve_settings(terminator: Any, meta: CompletionMeta | None) -> WatcherSettings:
    """YAML values where given, else the checkpoint's own."""

    def pick(value: float | None, default: float | None) -> float | None:
        return default if value is None else float(value)

    return WatcherSettings(
        threshold=pick(terminator.threshold, meta and meta.threshold),
        hold_s=pick(terminator.hold_s, meta and meta.hold_s),
        min_s=pick(terminator.min_s, meta and meta.min_s),
        mode=terminator.mode,
        rule=terminator.type,
        wait_s=terminator.wait_s,
    )


class CompletionWatcher:
    """Watches a stage's frames and grippers and ends the stage when it is done.

    ``offer`` runs on the control thread (inside predict_action). It updates the
    gripper tracker and, for a grasp stage, fires there; for the classifier it
    does at most a resize, and the network runs on the watcher thread, on its
    own CUDA stream, so a classifier pass does not queue behind -- or in front
    of -- the policy's own kernels on the default stream.

    Nothing here may take the control loop down. A failure in scoring is logged,
    recorded in the summary, and the watcher stops; the stage then ends on its
    timer, which is exactly what it did before classifiers existed.
    """

    def __init__(
        self,
        model: CompletionModel | None,
        settings: WatcherSettings,
        events: dict[str, bool],
        clock: Callable[[], float] = time.perf_counter,
        grip: GripTracker | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.events = events
        self.clock = clock
        self.grip = grip
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: tuple[float, list[torch.Tensor], Any, bool] | None = None
        self._last_offer_t: float | None = None
        self._started_at: float | None = None
        self._above_since: float | None = None
        self.samples: list[tuple[float, float]] = []
        self.offered = 0
        self.decided_at_s: float | None = None
        self.decided_score: float | None = None
        # First time the classifier would have decided but the gripper gate was
        # not satisfied (classifier stages with `grip` only).
        self.gate_blocked_at_s: float | None = None
        self.grasp_decided_at_s: float | None = None
        self.grasp_at_s: float | None = None
        self.fired = False
        self.skipped_reason = ""
        self.error = ""
        self.score_ms: list[float] = []
        cuda = model is not None and model.device.type == "cuda"
        self._stream = torch.cuda.Stream(device=model.device) if cuda else None

    @property
    def path(self) -> str:
        return self.model.path if self.model is not None else ""

    # --- control thread -------------------------------------------------------

    def start(self) -> None:
        self._started_at = self.clock()
        if self.model is None:
            return
        self._thread = threading.Thread(
            target=self._run, name="stage-completion", daemon=True
        )
        self._thread.start()

    def offer(self, observation: Mapping[str, Any]) -> None:
        """Hand the watcher this tick's observation.

        The grippers are read every tick, the frames at most once per period_s.
        Swallows every exception: it runs inside predict_action, and a raise
        here would end the policy stage with a traceback.
        """
        if self._started_at is None or self._stop.is_set():
            return
        now = self.clock()
        t = now - self._started_at
        holding = True
        if self.grip is not None:
            try:
                self.grip.update(t, observation[OBS_STATE_KEY].reshape(-1).tolist())
            except Exception as error:  # no observation.state, a device error
                self._fail(f"could not read the grippers: {error!r}")
                return
            if self.settings.rule == TERMINATOR_GRASP:
                self.decide_grasp(t)
            holding = self.grip.holding(t)
        if self.model is None:
            return
        if (
            self._last_offer_t is not None
            and now - self._last_offer_t < self.settings.period_s
        ):
            return
        self._last_offer_t = now
        try:
            prepared = self.model.prepare(observation)
            ready = None
            if self._stream is not None:
                ready = torch.cuda.Event()
                ready.record()
        except Exception as error:  # a missing camera key, a device error
            self._fail(f"could not prepare the frame: {error!r}")
            return
        with self._lock:
            self._pending = (t, prepared, ready, holding)
        self.offered += 1
        self._wake.set()

    def decide_grasp(self, t: float) -> None:
        """The grasp rule at t seconds into the stage. Public for the tests."""
        if self.grasp_decided_at_s is not None or self.grip is None:
            return
        grasp = self.grip.latest_grasp_s()
        if grasp is None or self.settings.wait_s is None:
            return
        if t - grasp < self.settings.wait_s:
            return
        self.grasp_decided_at_s = t
        self.grasp_at_s = grasp
        self._fire()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # --- watcher thread -------------------------------------------------------

    def _run(self) -> None:
        assert self.model is not None
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is None:
                continue
            t, prepared, ready, holding = pending
            try:
                began = time.perf_counter()
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        self._stream.wait_event(ready)
                        score = self.model.score(prepared)
                else:
                    score = self.model.score(prepared)
                self.score_ms.append((time.perf_counter() - began) * 1000.0)
            except Exception as error:
                self._fail(f"scoring failed: {error!r}")
                return
            self.decide(t, score, holding)

    def decide(self, t: float, score: float, holding: bool = True) -> None:
        """One scored frame at t seconds into the stage. Public for the tests.

        ``holding`` is the gripper gate at the moment the frame was offered
        (always True without a `grip` block).
        """
        self.samples.append((t, score))
        settings = self.settings
        if self.decided_at_s is not None:
            return
        if score < settings.threshold:
            self._above_since = None
            return
        if self._above_since is None:
            self._above_since = t
        if t < settings.min_s or t - self._above_since < settings.hold_s:
            return
        if not holding:
            # The score keeps its hold: the decision comes the moment the
            # grippers settle, not hold_s after.
            if self.gate_blocked_at_s is None:
                self.gate_blocked_at_s = t
            return
        self.decided_at_s = t
        self.decided_score = score
        if settings.rule == TERMINATOR_CLASSIFIER:
            self._fire()

    def _fire(self) -> None:
        if self.settings.mode != COMPLETION_MODE_FIRE or self.fired:
            return
        # exit_early already set means someone else is ending this loop -- the
        # right arrow, or the space bar handing over to the leader arm. The
        # stage is ending anyway; recording it as ours would mislabel it.
        if self.events.get("exit_early"):
            self.skipped_reason = "exit_early was already set"
            return
        self.events["exit_early"] = True
        self.fired = True

    def _fail(self, message: str) -> None:
        if not self.error:
            self.error = message
            logger.error(
                f"completion watcher {self.path or self.settings.rule}: {message}; "
                "the stage now ends on its timer"
            )
        self._stop.set()

    # --- after the stage ------------------------------------------------------

    def describe_fire(self) -> str:
        """The stage_end reason when this watcher ended the stage."""
        if self.settings.rule == TERMINATOR_GRASP:
            return (
                f"grasp at {self.grasp_at_s:.2f}s + wait {self.settings.wait_s:.2f}s, "
                f"decided at {self.grasp_decided_at_s:.2f}s"
            )
        gate = "" if self.grip is None else ", grippers holding"
        return (
            f"completion score {self.decided_score:.3f} >= "
            f"{self.settings.threshold:.3f} for {self.settings.hold_s:.2f}s{gate}, "
            f"decided at {self.decided_at_s:.2f}s"
        )

    def summary(self) -> dict[str, Any]:
        """The stage_completion event's fields. Raw samples, no metric."""
        scores = [score for _, score in self.samples]
        out = {
            "model_path": self.path,
            "rule": self.settings.rule,
            "mode": self.settings.mode,
            "threshold": self.settings.threshold,
            "hold_s": self.settings.hold_s,
            "min_s": self.settings.min_s,
            "period_s": self.settings.period_s,
            "fired": self.fired,
            "decided_at_s": self.decided_at_s,
            "decided_score": self.decided_score,
            "skipped_reason": self.skipped_reason,
            "error": self.error,
            "frames_offered": self.offered,
            "frames_scored": len(self.samples),
            "max_score": max(scores) if scores else None,
            "score_ms_median": _median(self.score_ms),
            "samples": [[round(t, 3), round(score, 4)] for t, score in self.samples],
        }
        if self.grip is not None:
            out |= {
                "grip_state_names": self.grip.names,
                "gate_blocked_at_s": self.gate_blocked_at_s,
                "wait_s": self.settings.wait_s,
                "grasp_at_s": self.grasp_at_s,
                "grasp_decided_at_s": self.grasp_decided_at_s,
                "grip_events": self.grip.events,
                # [t, width of each named gripper ...], about every period_s.
                "grip_samples": self.grip.samples,
            }
        return out


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


class TappedPreprocessor:
    """A policy preprocessor that shows each observation to a watcher first.

    Everything but the call is forwarded, so record_loop's
    ``preprocessor.reset()`` at loop entry still reaches the real pipeline.
    """

    def __init__(self, inner: Any, watcher: CompletionWatcher) -> None:
        self._inner = inner
        self._watcher = watcher

    def __call__(self, observation: Any) -> Any:
        self._watcher.offer(observation)
        return self._inner(observation)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def load_completion_models(
    stages: list[Any], device: str = "cuda"
) -> dict[str, CompletionModel]:
    """One CompletionModel per stage with a model_path, loaded once per path."""
    by_path: dict[str, CompletionModel] = {}
    models: dict[str, CompletionModel] = {}
    for stage in stages:
        if stage.terminator.type not in WATCHED_TERMINATORS:
            continue
        path = stage.terminator.model_path
        if not path:  # a grasp stage scoring no classifier
            continue
        if path not in by_path:
            by_path[path] = CompletionModel.load(path, device)
            logger.info(
                f"completion classifier for stage {stage.id!r}: {path} "
                f"(cameras {list(by_path[path].meta.camera_keys)})"
            )
        models[stage.id] = by_path[path]
    return models
