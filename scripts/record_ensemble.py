#!/usr/bin/env python
"""``lerobot-record`` with temporal ensembling bolted onto SmolVLA.

SmolVLA executes a whole action chunk open-loop (``n_action_steps=50``, i.e. 1.67 s at
30 Hz) and only then re-queries. Consecutive chunks disagree at the seam, and the arm
jitters there. ACT solves this by re-querying every step and averaging the overlapping
predictions (Zhao et al. 2023, Algorithm 2), but lerobot 0.4.4 only wires that into ACT --
``temporal_ensemble_coeff`` does nothing on SmolVLA.

This script reuses lerobot's own ``ACTTemporalEnsembler`` (it is policy-agnostic) and swaps
SmolVLA's queue-based ``select_action`` for an ensembling one, then hands over to the stock
record entry point. Nothing in lerobot or in the checkpoint is modified.

Why the inference runs off the control thread
---------------------------------------------
``record_loop`` calls the policy synchronously, so every millisecond of inference lands on the
control period. SmolVLA needs ~170 ms on the robot PC against a 33 ms budget at 30 Hz, and on
this platform a slow loop is not merely "slow": ``send_action`` holds the base velocity command
until the next call, so the loop period *is* the integration window that turns a velocity
command into rotation. Ensembling every step drops the loop to ~4.6 Hz against a ~21.5 Hz
recording baseline -- a 4.7x base over-rotation, which on a driving task swamps whatever the
seam jitter was costing.

So by default the chunk prediction runs on a private thread and a private CUDA stream
(``--ensemble.async=false`` restores the old synchronous path for an A/B). Two details make
that work rather than merely look like it works:

* **Private CUDA stream.** lerobot's ``predict_action`` ends in a ``.cpu()`` on the action.
  On the default stream that call waits for every kernel queued before it -- including the
  170 ms policy forward launched from the worker -- so a thread alone buys nothing.
* **Latency-aligned merge.** A chunk predicted from the observation at step ``t`` only lands
  at ``t + k``. Merging it head-first blends actions meant for ``t`` into the slot for
  ``t + k``, which injects error at exactly the seam ensembling is there to smooth. The
  worker's chunk is therefore advanced by the ``k`` steps that elapsed while it was in flight
  (``--ensemble.align=false`` to A/B that alone).

Usage (same arguments as lerobot-record, plus a few)::

    uv run python scripts/record_ensemble.py \
        --ensemble.coeff=0.01 \
        --robot.type=mobileai_robot ... --policy.path=...

``--ensemble.every=N`` predicts once every N control steps instead of every step, blending the
overlapping chunks in between. It is a synchronous-mode knob: there, ``every=1`` is what makes
plain ensembling unusable, and ``every=10`` matches the inference load of ``n_action_steps=10``
while still averaging the overlap. **Asynchronously, leave it at 1.** The control loop no longer
pays for inference, so ``every=1`` just means "re-query as fast as the GPU allows", and raising
it only buys back GPU time that was not on the critical path -- measured against a stub whose
chunks disagree by sigma=0.1, ``every=20`` async lands on the same RMS (0.074 vs 0.071) and the
same worst step-to-step jump (0.206) as ``every=20`` *synchronous*, against 0.040 / 0.143 at
``every=1``. Raise it only if the GPU is genuinely needed elsewhere.

``--ensemble.plugin=<module>`` force-imports a module that registers a robot/teleop type, for
setups where lerobot's ``lerobot_robot_*`` name-scan does not find it.

``--ensemble.coeff`` follows ACT's convention: positive weighs *older* predictions more
(0.01 is the published default), 0 weighs them equally, negative favours the newest. Heavily
favouring new predictions undoes the point of action chunking.

The exit line reports measured inference time and, asynchronously, the in-flight lag in control
steps -- that lag is ``k``, and it is the number to watch. Read the achieved control rate off
the ``Control loop rate`` summary and divide the recording run's by the eval run's to get the
base over-rotation multiplier.
"""

import logging
import queue
import sys
import threading
import time
from pathlib import Path

# 콘솔 스크립트(lerobot-record)로 실행할 때와 달리 `python scripts/record_ensemble.py` 는
# sys.path[0] 이 scripts/ 라 프로젝트 루트가 안 잡힌다. 0.4.4 의 플러그인 탐색은 sys.path 가
# 아니라 importlib.metadata 를 쓰므로 --robot.type 은 그것만으로 해결되지만, 루트 기준 임포트를
# 쓰는 --ensemble.plugin 과 구버전 lerobot 을 위해 루트를 먼저 얹어 둔다.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging

try:  # lerobot >= 0.4.2
    from lerobot.utils.import_utils import register_third_party_plugins
except ImportError:  # lerobot 0.4.0/0.4.1 spelling
    from lerobot.utils.import_utils import (
        register_third_party_devices as register_third_party_plugins,
    )
import lerobot.scripts.lerobot_record as lerobot_record


def _pop_arg(name: str, default: str) -> str:
    """Remove ``--name=value`` (or ``--name value``) from argv; draccus rejects unknowns."""
    for i, arg in enumerate(sys.argv):
        if arg == name and i + 1 < len(sys.argv):
            value = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            return value
        if arg.startswith(f"{name}="):
            del sys.argv[i]
            return arg.split("=", 1)[1]
    return default


def _pop_flag(name: str, default: bool) -> bool:
    raw = _pop_arg(name, "true" if default else "false").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} 은 true/false 여야 한다 (받은 값: {raw!r})")


class _Stats:
    __slots__ = ("calls", "total_s", "worst_s", "lag_total", "lag_worst", "dropped")

    def __init__(self):
        self.calls = 0
        self.total_s = 0.0
        self.worst_s = 0.0
        self.lag_total = 0
        self.lag_worst = 0
        self.dropped = 0

    def observe(self, elapsed_s: float, lag_steps: int = 0) -> None:
        self.calls += 1
        self.total_s += elapsed_s
        self.worst_s = max(self.worst_s, elapsed_s)
        self.lag_total += lag_steps
        self.lag_worst = max(self.lag_worst, lag_steps)

    def report(self) -> str:
        if not self.calls:
            return "정책 추론이 한 번도 호출되지 않았다"
        avg = self.total_s / self.calls
        line = (
            f"추론 {self.calls}회 · 평균 {avg * 1000:.1f} ms (최악 {self.worst_s * 1000:.1f} ms)"
            f" -> 동기로 돌렸다면 지속 가능한 제어 주파수 약 {1 / avg:.0f} Hz"
        )
        if self.lag_total or self.lag_worst:
            line += (
                f" · 비행 중 지연 평균 {self.lag_total / self.calls:.1f} 스텝"
                f" (최악 {self.lag_worst})"
            )
        if self.dropped:
            line += f" · 너무 늦어 버린 청크 {self.dropped}개"
        return line


def _merge(ens: ACTTemporalEnsembler, actions: torch.Tensor) -> None:
    """Fold a freshly predicted chunk into the running ensemble.

    ``ACTTemporalEnsembler.update`` assumes exactly one action was consumed since the last
    call, so its shapes line up as (chunk-1) against (chunk-1). Predicting every N steps
    leaves the buffer N shorter than that, and latency alignment trims the head off the new
    chunk, so either side can be the longer one -- hence this generalised merge over the
    overlap. Both start at the same timestep, so they align at the head.
    """
    ens.ensemble_weights = ens.ensemble_weights.to(device=actions.device)
    ens.ensemble_weights_cumsum = ens.ensemble_weights_cumsum.to(device=actions.device)

    if ens.ensembled_actions is None:
        ens.ensembled_actions = actions.clone()
        ens.ensembled_actions_count = torch.ones(
            (actions.shape[1], 1), dtype=torch.long, device=actions.device
        )
        return

    buf_len = ens.ensembled_actions.shape[1]
    new_len = actions.shape[1]
    overlap = min(buf_len, new_len)

    count = ens.ensembled_actions_count[:overlap]
    head = ens.ensembled_actions[:, :overlap]
    head = head * ens.ensemble_weights_cumsum[count - 1]
    head = head + actions[:, :overlap] * ens.ensemble_weights[count]
    head = head / ens.ensemble_weights_cumsum[count]
    head_count = torch.clamp(count + 1, max=ens.chunk_size)

    if new_len > overlap:
        # The new chunk reaches further ahead than the buffer: its tail is unaveraged.
        tail = actions[:, overlap:]
        tail_count = torch.ones((new_len - overlap, 1), dtype=torch.long, device=actions.device)
    else:
        # The buffer reaches further than the new chunk (a head-trimmed chunk does this).
        # Keep the buffer's tail rather than truncating the horizon to the new chunk.
        tail = ens.ensembled_actions[:, overlap:]
        tail_count = ens.ensembled_actions_count[overlap:]

    ens.ensembled_actions = torch.cat([head, tail], dim=1)
    ens.ensembled_actions_count = torch.cat([head_count, tail_count])


def _pop(ens: ACTTemporalEnsembler) -> torch.Tensor:
    """Consume the action for the current timestep."""
    action = ens.ensembled_actions[:, 0]
    ens.ensembled_actions = ens.ensembled_actions[:, 1:]
    ens.ensembled_actions_count = ens.ensembled_actions_count[1:]
    return action


class _Worker:
    """Predicts one chunk at a time on a private thread and a private CUDA stream.

    The stream is the load-bearing half. CUDA work launched from any thread lands on the
    default stream unless told otherwise, and ``predict_action`` finishes each control step
    with a ``.cpu()`` -- which would then wait out the policy forward this worker just queued.
    A private stream lets the control thread's own (tiny) GPU work overtake it.

    ``generation`` invalidates whatever is in flight when an episode resets: the result still
    arrives, and is dropped on arrival.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._requests: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._generation = 0
        self._inflight = 0
        thread = threading.Thread(target=self._run, name="ensemble-inference", daemon=True)
        thread.start()

    @property
    def busy(self) -> bool:
        return self._inflight > 0

    def _run(self) -> None:
        stream = None
        if self._device.type == "cuda":
            # A fresh thread starts on cuda:0 whatever the policy is on; pin it first so the
            # stream and the forward land on the same device.
            torch.cuda.set_device(self._device)
            stream = torch.cuda.Stream(device=self._device)
        while True:
            fn, batch, step, generation = self._requests.get()
            started = time.perf_counter()
            actions, error = None, None
            try:
                # The control thread runs under inference_mode; match it so the tensors this
                # produces can be merged there without crossing modes.
                with torch.inference_mode():
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            actions = fn(batch)
                        # Hand over only settled memory, and time the GPU rather than the launch.
                        stream.synchronize()
                    else:
                        actions = fn(batch)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the control thread
                error = exc
            self._results.put((actions, step, generation, time.perf_counter() - started, error))

    def submit(self, fn, batch, step: int) -> None:
        self._inflight += 1
        self._requests.put((fn, batch, step, self._generation))

    def poll(self):
        """Return a landed chunk, or None if nothing landed (or it landed stale)."""
        try:
            return self._accept(self._results.get_nowait())
        except queue.Empty:
            return None

    def wait(self):
        """Block for the next result. Returns None if it turns out to be stale."""
        return self._accept(self._results.get())

    def _accept(self, item):
        actions, step, generation, elapsed, error = item
        self._inflight = max(0, self._inflight - 1)
        if error is not None:
            raise error
        if generation != self._generation:
            return None
        return actions, step, elapsed

    def flush(self) -> None:
        """Invalidate everything in flight -- called when an episode resets.

        A request the worker already picked up is still running; its result arrives later and
        the generation check drops it. ``_inflight`` deliberately keeps counting it, so ``busy``
        stays true meanwhile -- zeroing it here would let the new episode queue a second request
        behind the stale forward, burning an extra ~170 ms at every episode start and leaving
        the count short for the rest of the run.
        """
        self._generation += 1
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                break
            else:
                self._inflight = max(0, self._inflight - 1)


class _State:
    __slots__ = ("ens", "step", "last_submit", "worker")

    def __init__(self, ens: ACTTemporalEnsembler):
        self.ens = ens
        self.step = 0
        # Far enough back that the first step is always due.
        self.last_submit = -(1 << 30)
        self.worker: _Worker | None = None


def install(coeff: float, every: int = 1, use_async: bool = True, align: bool = True) -> _Stats:
    """Replace SmolVLA's chunk queue with a temporal ensembler."""
    stats = _Stats()
    original_reset = SmolVLAPolicy.reset

    def _state(self) -> _State:
        state = getattr(self, "_ens_state", None)
        if state is None:
            state = _State(ACTTemporalEnsembler(coeff, self.config.chunk_size))
            state.ens.reset()
            self._ens_state = state
        return state

    def _fold(self, state: _State, landed) -> None:
        """Align a landed chunk to the current timestep and merge it."""
        actions, request_step, elapsed = landed
        lag = state.step - request_step if align else 0
        stats.observe(elapsed, state.step - request_step)

        # The ensembler is sized for chunk_size; a policy returning more would silently
        # misalign the weights, so keep only what it expects.
        chunk = self.config.chunk_size
        if actions.shape[1] > chunk:
            actions = actions[:, :chunk]
        if lag >= actions.shape[1]:
            stats.dropped += 1
            return
        if lag > 0:
            # actions[:, j] is the prediction for timestep request_step + j; the buffer head
            # is timestep state.step. Drop the j < lag entries that are already in the past.
            actions = actions[:, lag:]
        _merge(state.ens, actions)

    def reset(self):
        original_reset(self)
        state = _state(self)
        state.ens = ACTTemporalEnsembler(coeff, self.config.chunk_size)
        state.ens.reset()
        state.step = 0
        state.last_submit = -(1 << 30)
        if state.worker is not None:
            state.worker.flush()

    @torch.no_grad()
    def select_action(self, batch, noise=None):
        # Same preamble as the original: normalise inputs and refresh the observation queues.
        if self.training:
            # Upstream calls eval() unconditionally; make it conditional so the control thread
            # never writes module flags while the worker is mid-forward.
            self.eval()
        batch = self._prepare_batch(batch)
        from lerobot.policies.utils import populate_queues

        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        state = _state(self)
        ens = state.ens

        def dry() -> bool:
            return ens.ensembled_actions is None or ens.ensembled_actions.shape[1] == 0

        if not use_async:
            if dry() or state.step - state.last_submit >= every:
                started = time.perf_counter()
                actions = self._get_action_chunk(batch, noise)  # (batch, chunk, action_dim)
                # Nothing was consumed while the loop blocked, so the chunk is still aligned.
                _fold(self, state, (actions, state.step, time.perf_counter() - started))
                state.last_submit = state.step
            state.step += 1
            return _pop(ens)

        if state.worker is None:
            state.worker = _Worker(next(self.parameters()).device)
        worker = state.worker

        # 1. Fold in whatever finished while the robot was being driven.
        landed = worker.poll()
        if landed is not None:
            _fold(self, state, landed)

        # 2. Queue the next prediction. `dict(batch)` because _get_action_chunk mutates it.
        if not worker.busy and state.step - state.last_submit >= every:
            worker.submit(self._get_action_chunk, dict(batch), state.step)
            state.last_submit = state.step

        # 3. Only ever block when there is genuinely nothing to send -- the first step of an
        #    episode, or a stall long enough to drain a whole chunk.
        while dry():
            if not worker.busy:
                worker.submit(self._get_action_chunk, dict(batch), state.step)
                state.last_submit = state.step
            landed = worker.wait()
            if landed is not None:
                _fold(self, state, landed)

        state.step += 1
        return _pop(ens)

    SmolVLAPolicy.reset = reset
    SmolVLAPolicy.select_action = select_action
    return stats


def _report_registered() -> None:
    """Log which robot/teleop types the parser will accept, so a missing plugin is obvious."""
    try:
        from lerobot.robots.config import RobotConfig
        from lerobot.teleoperators.config import TeleoperatorConfig

        for label, cls in (("robot", RobotConfig), ("teleop", TeleoperatorConfig)):
            names = sorted(cls.get_known_choices())
            logging.info(f"등록된 {label}.type: {', '.join(names) if names else '(없음)'}")
    except Exception as exc:  # pragma: no cover - 진단용이라 실패해도 진행한다
        logging.debug(f"등록 목록을 읽지 못했다: {exc}")


def main() -> None:
    coeff = float(_pop_arg("--ensemble.coeff", "0.01"))
    every = max(1, int(_pop_arg("--ensemble.every", "1")))
    use_async = _pop_flag("--ensemble.async", True)
    align = _pop_flag("--ensemble.align", True)
    extra = _pop_arg("--ensemble.plugin", "")
    init_logging()

    # main() 안에서도 호출되지만, 여기서 먼저 부르고 결과를 찍어 둔다 -- 플러그인이 안 잡힌 채
    # 파서가 돌면 "invalid choice: 'mobileai_robot'" 만 나와서 원인을 알기 어렵다.
    register_third_party_plugins()
    if extra:
        import importlib

        importlib.import_module(extra)
        logging.info(f"추가 플러그인 임포트: {extra}")
    _report_registered()

    stats = install(coeff, every, use_async, align)
    cadence = "매 제어 스텝마다" if every == 1 else f"{every} 스텝마다"
    if use_async:
        logging.info(
            f"temporal ensembling 활성화 (coeff={coeff}, every={every}, async, "
            f"align={'on' if align else 'off'}). 추론은 전용 스레드·CUDA 스트림에서 돌고 제어 "
            f"루프는 안 멈춘다. {cadence} 새 청크를 요청하고, 도착한 청크는 비행 중 흘러간 "
            "스텝만큼 앞을 잘라 맞춘 뒤 겹치는 구간을 가중 평균한다."
        )
    else:
        logging.warning(
            f"temporal ensembling 활성화 (coeff={coeff}, every={every}, 동기). 정책을 {cadence} "
            "호출하고 그동안 제어 루프가 멈춘다 -- 이 로봇에서 루프 저하는 그대로 base 과회전이다. "
            "A/B 목적이 아니라면 --ensemble.async=true 로 돌릴 것."
        )
    try:
        lerobot_record.main()
    finally:
        logging.info(stats.report())


if __name__ == "__main__":
    main()
