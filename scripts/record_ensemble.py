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

``--ensemble.amp=bf16`` runs the chunk forward under autocast. It exists because
``--policy.use_amp=true`` is a **silent no-op here**: lerobot enables autocast inside
``predict_action`` on the control thread, and autocast state is thread-local, so it never
reaches the worker's forward. Whether it actually pays is an open question on SmolVLA -- the
backbone re-casts activations to the weight dtype at every layer and forces attention math to
fp32 (``smolvlm_with_expert.py:222,307,528``), so the GEMMs speed up but the cast traffic
grows. Read the exit line's mean inference time with it on and off. ``bf16`` rather than
``fp16`` by default when enabled: lerobot's own autocast has no dtype and so picks fp16, whose
overflow surfaces as ``Joint 0 position input contains NaN`` -- which this repo's README
attributes to corrupt normalizer stats, so it would be misdiagnosed.

``--ensemble.noise_scale=S`` and ``--ensemble.samples=K`` act on the other source of seam
jitter: SmolVLA is a flow-matching sampler, so two predictions from the *same* observation
differ by whatever initial noise each started from (measured here: RMS 0.27 in normalised
action units, against 0.003-0.009 for bf16 rounding). Temporal ensembling averages chunks
across time; these two tame the sample itself, and cost no extra re-queries:

* ``noise_scale`` multiplies the initial Gaussian noise. 1.0 is the trained distribution,
  0.5 measured a 0.27 -> 0.09 drop in chunk-to-chunk disagreement for free, 0 makes the
  policy deterministic (same observation, same chunk -- like ACT) at the risk of blander
  motion, since the sampler then starts somewhere training never did.
* ``samples`` draws K noise samples as one batch and averages the K chunks. K=4 measured
  0.27 -> 0.12 for +70 ms (bf16, 20 denoising steps); the batch shares one forward, so the
  cost is far below K-fold. Averaging is only safe where the policy's samples agree on
  *what* to do and differ in detail -- the mean of "go left" and "go right" is neither.

Both default to the stock behaviour (1.0, 1) and work in either mode.

``--ensemble.merge=latest`` turns the temporal averaging off without giving up asynchronous
re-querying: a chunk that lands simply *replaces* whatever is left of the previous one (after
the same latency trim), which is lerobot's ``latest_only`` aggregate. The default,
``average``, is the ACT-style weighted mean governed by ``--ensemble.coeff``. Do not reach
for a very negative ``coeff`` to approximate this -- below about -1.7 the float32 weights
``exp(-coeff * i)`` overflow to inf.

``--ensemble.commit=N`` decouples the two rates. The worker keeps predicting back to back
from the newest observation (so the GPU never idles), but the robot *commits* to one chunk
for N control steps and only then switches -- to the freshest chunk that has landed, trimmed
by the steps that elapsed since it was requested. Chunks that land in between are superseded
and never executed. This is SmolVLA's stock "run n_action_steps, then re-plan" behaviour
minus the stall at the boundary. N cannot usefully exceed ``chunk_size - lag``: a chunk that
is taken over `lag` steps after its observation has only that many actions left, and the
switch happens early when it runs out. Asynchronous mode only; leave ``every`` at 1.

What happens to the chunks that land *during* a commit window depends on ``--ensemble.merge``.
With ``average`` (the default) they are not thrown away: each is trimmed to the current
timestep and folded into a pending ensemble (same ``--ensemble.coeff`` weighting), so the
chunk the robot switches to is the mean of the ~N/lag predictions made while it was busy --
sampling noise down by roughly the square root of that count, from GPU time that was being
spent anyway. The switch itself is then a linear cross-fade over whatever is left of the old
chunk rather than a cut, so a smaller N buys a longer fade (N=30 leaves ~12 steps). With
``latest`` only the newest landed chunk is kept and the switch is a hard replacement.

``--ensemble.fade=F`` bounds that cross-fade to F steps (0, the default, keeps the behaviour
above). It matters at both ends of N: at ``commit=50`` the chunk runs dry at ``50 - lag`` and the
switch has nothing left to fade over -- measured offline at up to 0.64 rad per step on one joint --
while at a small N the whole-remainder fade keeps the robot on the old plan for most of the window.
``--ensemble.commit=30 --ensemble.fade=5`` kept the seam under ~0.13 rad and completed as often as
``commit=50`` in the closed-loop replay (docs/offline_eval_2026-09-22.md).

``--ensemble.reduce=medoid`` returns the one sample (of ``--ensemble.samples``) closest to the
others instead of their mean. Tested offline against the mean at noise 1.0: no gain, jerkier.

``--ensemble.base_rate_hz=H`` scales the base velocity command by ``measured loop Hz / H`` when the
loop runs slower than the recording did (H = 21.5 on this robot), so each control step drives the
base as far as a demonstration step did. Off by default; not yet tried on the robot.

``--ensemble.worker=process`` (the default) runs the asynchronous forward in a *separate
process* instead of a thread. On this robot a thread cannot work: the Mobile AI base driver
(``trossen_slate``, pybind11) holds the GIL for the whole of ``update_state()`` and
``set_cmd_vel()`` -- measured at 25 ms per call with every other Python thread frozen for
the duration -- and the control loop spends 42 of its 48 ms per iteration inside those two
calls. A worker thread therefore gets ~6 ms of Python in every 48, which is the 8.0x launch
slowdown measured on the robot (146.7 ms synchronous vs 1173 ms on the thread, GPU idle
throughout). A process has its own GIL. It loads its own copy of the policy (~1.2 GB) and
receives each observation through shared memory; ``--ensemble.worker=thread`` keeps the old
path for an A/B.

``--ensemble.profile=true`` splits each measured inference into queue wait, GPU time (CUDA
events recorded on the worker's own stream) and the CPU-side remainder, and prints the split
on the exit line. It answers the one question the wall-clock number cannot: whether a slow
forward is the GPU genuinely working that long, or the worker thread being kept off the CPU
between kernel launches. Off by default -- it adds two CUDA events per forward.

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

import atexit
import contextlib
import functools
import logging
import os
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


def _amp(mode: str, device: torch.device):
    """A fresh autocast context for one forward, or a no-op.

    Nested inside ``predict_action``'s autocast (when ``--policy.use_amp=true``) this wins --
    the inner context overrides the outer dtype, verified -- so the flag means the same thing
    on the control thread as it does on the worker.
    """
    if mode == "off" or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16
    )


def _reduce_samples(actions: torch.Tensor, reduce: str) -> torch.Tensor:
    """(K, chunk, dim) -> (1, chunk, dim).

    ``mean`` averages the K chunks. ``medoid`` returns the one real sample whose summed distance
    to the other K-1 is smallest: it rejects an outlier draw the way the mean does, but it does not
    shrink the motion -- the mean of K draws that disagree on *when* or *how far* to move is a
    chunk that moves less than any of them (measured offline on task02/task06: see
    docs/offline_eval_2026-09-22.md).
    """
    if actions.shape[0] == 1:
        return actions
    if reduce == "mean":
        return actions.mean(dim=0, keepdim=True)
    if reduce == "medoid":
        flat = actions.reshape(actions.shape[0], -1).float()
        i = torch.cdist(flat, flat).sum(dim=1).argmin()
        return actions[i : i + 1]
    raise ValueError(f"unknown reduce {reduce!r}")


def _predict_chunk(
    policy, batch, noise, samples: int, noise_scale: float, reduce: str = "mean"
) -> torch.Tensor:
    """One chunk, from `samples` noise draws scaled by `noise_scale`, reduced by `reduce`.

    Module-level so the worker process can call exactly what the control thread calls.
    """
    if noise is not None or (samples == 1 and noise_scale == 1.0):
        return policy._get_action_chunk(batch, noise)  # stock path, untouched
    if samples > 1:
        # One observation, K rows: the K samples share a single batched forward.
        batch = {
            k: v.expand(samples, *v.shape[1:]) if torch.is_tensor(v) and v.shape[0] == 1 else v
            for k, v in batch.items()
        }
    cfg = policy.config
    device = next(policy.parameters()).device
    noise = torch.randn(
        samples, cfg.chunk_size, cfg.max_action_dim, dtype=torch.float32, device=device
    )
    if noise_scale != 1.0:
        noise = noise * noise_scale
    actions = policy._get_action_chunk(batch, noise)  # (samples, chunk, action_dim)
    return _reduce_samples(actions, reduce)


def _proc_main(
    requests, results, config, path, device_str, amp, samples, noise_scale, reduce="mean"
) -> None:
    """Worker process: own interpreter, own GIL, own copy of the policy."""
    try:
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        policy = SmolVLAPolicy.from_pretrained(path, config=config).to(device).eval()
        shared: dict = {}
        results.put(("ready", None))
    except BaseException as exc:  # noqa: BLE001
        results.put(("fatal", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        msg = requests.get()
        if msg is None:
            return
        kind = msg[0]
        if kind == "buffers":  # the shared-memory tensors, sent once (again if shapes change)
            shared = msg[1]
            continue
        _, extras, step, generation, queued_at = msg
        started = time.perf_counter()
        actions, error = None, None
        try:
            batch = {k: v.to(device, non_blocking=False) for k, v in shared.items()}
            batch.update(extras)
            t_h2d = time.perf_counter()
            with torch.inference_mode(), _amp(amp, device):
                out = _predict_chunk(policy, batch, None, samples, noise_scale, reduce)
            t_launch = time.perf_counter()
            actions = out.float().cpu()  # also the synchronisation point
            if os.environ.get("RECORD_ENSEMBLE_DEBUG"):
                print(
                    f"[worker] h2d {(t_h2d - started) * 1e3:.1f} ms | launch "
                    f"{(t_launch - t_h2d) * 1e3:.1f} ms | sync+d2h "
                    f"{(time.perf_counter() - t_launch) * 1e3:.1f} ms | threads "
                    f"{torch.get_num_threads()}",
                    file=sys.stderr, flush=True,
                )
        except BaseException as exc:  # noqa: BLE001 - re-raised on the control thread
            error = f"{type(exc).__name__}: {exc}"
        results.put(
            ("result", actions, step, generation, time.perf_counter() - started, error,
             (started - queued_at, None, None, None))
        )


class _ProcWorker:
    """Same surface as `_Worker`, but the forward runs in another process.

    Observations cross through shared-memory CPU tensors allocated once: per request the
    control thread pays one device-to-host copy of the batch and a tiny queue message. Only
    one request is ever in flight, and `busy` stays true until its result lands, so the
    buffers are never rewritten under a reader.
    """

    def __init__(self, policy, amp: str, samples: int, noise_scale: float, reduce: str = "mean"):
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")  # CUDA cannot survive a fork
        # Not next(policy.parameters()).device: this runs from the policy's constructor, when
        # the weights are still on the CPU. The config says where they are headed.
        self._policy = policy
        worker_device = torch.device(policy.config.device or "cpu")
        if worker_device.type == "cuda" and worker_device.index is None:
            worker_device = torch.device("cuda", torch.cuda.current_device())
        self._requests = ctx.Queue()
        self._results = ctx.Queue()
        self._generation = 0
        self._inflight = 0
        self._shared: dict = {}
        self._ready = False
        self.submit_total_s = 0.0
        self.submits = 0
        self._proc = ctx.Process(
            target=_proc_main,
            args=(self._requests, self._results, policy.config,
                  str(policy.config.pretrained_path), str(worker_device), amp, samples, noise_scale,
                  reduce),
            name="ensemble-inference",
            daemon=True,
        )
        self._proc.start()
        atexit.register(self.close)

    @property
    def busy(self) -> bool:
        return self._inflight > 0

    def _get(self, block: bool):
        """Next message, never hanging on a dead worker (the thread version could)."""
        while True:
            try:
                msg = self._results.get(timeout=1.0) if block else self._results.get_nowait()
            except queue.Empty:
                if not block:
                    return None
                if not self._proc.is_alive():
                    raise RuntimeError("추론 워커 프로세스가 죽었다 (위 로그의 트레이스백 참고)")
                continue
            if msg[0] == "ready":
                self._ready = True
                continue
            if msg[0] == "fatal":
                raise RuntimeError(f"추론 워커 프로세스 시작 실패: {msg[1]}")
            return msg

    def wait_ready(self) -> None:
        while not self._ready:
            if not self._proc.is_alive():
                self._get(block=False)
                raise RuntimeError("추론 워커 프로세스가 준비 전에 죽었다")
            try:
                msg = self._results.get(timeout=1.0)
            except queue.Empty:
                continue
            if msg[0] == "ready":
                self._ready = True
            elif msg[0] == "fatal":
                raise RuntimeError(f"추론 워커 프로세스 시작 실패: {msg[1]}")

    def submit(self, fn, batch, step: int) -> None:  # `fn` unused: the process owns the forward
        t0 = time.perf_counter()
        tensors = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        extras = {k: v for k, v in batch.items() if not torch.is_tensor(v)}
        stale = set(tensors) != set(self._shared) or any(
            self._shared[k].shape != v.shape or self._shared[k].dtype != v.dtype
            for k, v in tensors.items()
        )
        if stale:
            self._shared = {
                k: torch.empty(v.shape, dtype=v.dtype).share_memory_() for k, v in tensors.items()
            }
            self._requests.put(("buffers", self._shared))
        for k, v in tensors.items():
            self._shared[k].copy_(v)
        self._inflight += 1
        self._requests.put(("predict", extras, step, self._generation, t0))
        self.submit_total_s += time.perf_counter() - t0
        self.submits += 1

    def poll(self):
        msg = self._get(block=False)
        return None if msg is None else self._accept(msg)

    def wait(self):
        return self._accept(self._get(block=True))

    def _accept(self, msg):
        _, actions, step, generation, elapsed, error, prof = msg
        self._inflight = max(0, self._inflight - 1)
        if error is not None:
            raise RuntimeError(f"추론 워커 프로세스에서 예외: {error}")
        if generation != self._generation:
            return None
        return actions.to(next(self._policy.parameters()).device), step, elapsed, prof

    def flush(self) -> None:
        self._generation += 1
        while True:
            try:
                msg = self._results.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "result":
                self._inflight = max(0, self._inflight - 1)
            elif msg[0] == "ready":
                self._ready = True

    def close(self) -> None:
        if self._proc.is_alive():
            self._requests.put(None)
            self._proc.join(timeout=5.0)
            if self._proc.is_alive():
                self._proc.terminate()


class _Sampler:
    """Sample the worker thread's innermost frame while it is inside a forward.

    ``sys._current_frames()`` reads another live thread's stack from this process, so this
    needs neither py-spy nor ptrace. It answers the question a wall-clock split cannot: when
    the launch loop takes 1.2 s that the GPU finishes instantly, *which* Python frame is it
    sitting in.
    """

    def __init__(self, period_s: float = 0.005):
        self._period = period_s
        self._counts: dict = {}
        self._samples = 0
        self._tid: int | None = None
        self._active = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ensemble-sampler", daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def bind(self, tid: int) -> None:
        self._tid = tid

    def set_active(self, active: bool) -> None:
        self._active = active

    def _run(self) -> None:
        while not self._stop.wait(self._period):
            if not self._active or self._tid is None:
                continue
            frame = sys._current_frames().get(self._tid)
            if frame is None:
                continue
            code = frame.f_code
            key = f"{Path(code.co_filename).name}:{frame.f_lineno} {code.co_name}"
            self._counts[key] = self._counts.get(key, 0) + 1
            self._samples += 1

    def stop(self) -> None:
        self._stop.set()

    def report(self, top: int = 8) -> str:
        if not self._samples:
            return ""
        rows = sorted(self._counts.items(), key=lambda kv: -kv[1])[:top]
        body = "; ".join(f"{k} {v / self._samples:.0%}" for k, v in rows)
        return f"워커 스레드 샘플 {self._samples}회 -> {body}"


class _Stats:
    __slots__ = (
        "calls",
        "total_s",
        "worst_s",
        "lag_total",
        "lag_worst",
        "dropped",
        "gpu_total_s",
        "gpu_worst_s",
        "wait_total_s",
        "wait_worst_s",
        "launch_total_s",
        "launch_worst_s",
        "sync_total_s",
        "profiled",
        "sampler",
        "proc_worker",
        "superseded",
    )

    def __init__(self):
        self.calls = 0
        self.total_s = 0.0
        self.worst_s = 0.0
        self.lag_total = 0
        self.lag_worst = 0
        self.dropped = 0
        self.gpu_total_s = 0.0
        self.gpu_worst_s = 0.0
        self.wait_total_s = 0.0
        self.wait_worst_s = 0.0
        self.launch_total_s = 0.0
        self.launch_worst_s = 0.0
        self.sync_total_s = 0.0
        self.profiled = 0
        self.sampler = None
        self.proc_worker = None
        self.superseded = 0

    def observe(self, elapsed_s: float, lag_steps: int = 0, prof=None) -> None:
        self.calls += 1
        self.total_s += elapsed_s
        self.worst_s = max(self.worst_s, elapsed_s)
        self.lag_total += lag_steps
        self.lag_worst = max(self.lag_worst, lag_steps)
        if prof is not None:
            wait_s, gpu_s, launch_s, sync_s = prof
            self.profiled += 1
            self.wait_total_s += wait_s
            self.wait_worst_s = max(self.wait_worst_s, wait_s)
            if gpu_s is not None:
                self.gpu_total_s += gpu_s
                self.gpu_worst_s = max(self.gpu_worst_s, gpu_s)
            if launch_s is not None:
                self.launch_total_s += launch_s
                self.launch_worst_s = max(self.launch_worst_s, launch_s)
                self.sync_total_s += sync_s

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
        if self.superseded:
            line += f" · 실행 안 하고 넘긴 청크 {self.superseded}개 (commit)"
        if self.proc_worker is not None and self.proc_worker.submits:
            w = self.proc_worker
            line += (
                f" · 프로세스 워커: 제출(관측 복사) 평균 "
                f"{w.submit_total_s / w.submits * 1000:.1f} ms/회"
            )
        if self.profiled:
            # Wall time splits into three: waiting for the worker thread to pick the request
            # up, GPU time inside the forward, and whatever is left -- which is the CPU side
            # of the forward (kernel launches, Python) not overlapping the GPU.
            wall = self.total_s / self.calls
            wait = self.wait_total_s / self.profiled
            gpu = self.gpu_total_s / self.profiled if self.gpu_total_s else 0.0
            line += f" | 내역: 큐 대기 {wait * 1000:.1f} ms (최악 {self.wait_worst_s * 1000:.1f})"
            if self.gpu_total_s:  # thread worker only; the process worker reports no CUDA events
                line += (
                    f" · GPU {gpu * 1000:.1f} ms (최악 {self.gpu_worst_s * 1000:.1f})"
                    f" · 나머지(CPU 측) {(wall - wait - gpu) * 1000:.1f} ms"
                )
            if self.launch_total_s:
                launch = self.launch_total_s / self.profiled
                sync = self.sync_total_s / self.profiled
                line += (
                    f" | 런치 {launch * 1000:.1f} ms"
                    f" (최악 {self.launch_worst_s * 1000:.1f})"
                    f" · 동기화 대기 {sync * 1000:.1f} ms"
                )
            if self.sampler is not None:
                sampled = self.sampler.report()
                if sampled:
                    line += f"\n{sampled}"

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


def _crossfade(old: torch.Tensor | None, new: torch.Tensor, merge: str, fade: int) -> torch.Tensor:
    """commit-mode switch: blend the head of `new` in from what is left of `old`.

    ``fade=0`` keeps the original behaviour: under ``average`` the fade spans the *whole* overlap
    with the old chunk (``chunk - lag - commit`` steps, ~40 at commit=5), so a short commit spends
    most of its time executing the old plan; under ``latest`` the switch is a hard cut.
    ``fade=F>0`` fades over at most F steps under either merge -- the new chunk is fully in
    charge F steps after the switch, whatever N is.
    """
    if old is None or old.shape[1] == 0:
        return new
    if fade <= 0 and merge == "latest":
        return new
    n = min(old.shape[1], new.shape[1])
    if fade > 0:
        n = min(n, fade)
    w = torch.linspace(1 / (n + 1), n / (n + 1), n, device=new.device, dtype=new.dtype).view(1, n, 1)
    new = new.clone()
    new[:, :n] = (1 - w) * old[:, :n].to(new.dtype) + w * new[:, :n]
    return new


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

    def __init__(
        self,
        device: torch.device,
        amp: str = "off",
        profile: bool = False,
        sampler: "_Sampler | None" = None,
    ):
        self._device = device
        self._amp = amp
        self._profile = profile
        self._sampler = sampler
        self._requests: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._generation = 0
        self._inflight = 0
        self._thread = threading.Thread(target=self._run, name="ensemble-inference", daemon=True)
        self._thread.start()
        # A daemon thread parked in queue.get() is killed mid-C++-call when the interpreter
        # shuts down, which intermittently aborts with "terminate called without an active
        # exception" *after* a successful run -- a SIGABRT that reads like a crash. Ask it to
        # leave on its own instead.
        atexit.register(self.close)

    @property
    def busy(self) -> bool:
        return self._inflight > 0

    def _run(self) -> None:
        if self._sampler is not None:
            self._sampler.bind(threading.get_ident())
        stream = None
        if self._device.type == "cuda":
            # A fresh thread starts on cuda:0 whatever the policy is on; pin it first so the
            # stream and the forward land on the same device.
            torch.cuda.set_device(self._device)
            stream = torch.cuda.Stream(device=self._device)
        while True:
            request = self._requests.get()
            if request is None:  # shutdown sentinel
                return
            fn, batch, step, generation, queued_at = request
            started = time.perf_counter()
            actions, error = None, None
            gpu_s = launch_s = sync_s = None
            # How long the request sat in the queue before this thread got to it. Wall time
            # minus this minus GPU time is the CPU side of the forward, which is what
            # distinguishes "starved" from "the GPU really is this slow".
            wait_s = started - queued_at
            ev0 = ev1 = None
            if self._profile and stream is not None:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
            if self._sampler is not None:
                self._sampler.set_active(True)
            try:
                # The control thread runs under inference_mode; match it so the tensors this
                # produces can be merged there without crossing modes.
                with torch.inference_mode(), _amp(self._amp, self._device):
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            if ev0 is not None:
                                ev0.record(stream)
                            actions = fn(batch)
                            if ev1 is not None:
                                ev1.record(stream)
                        # fn() returns once every kernel is *launched*, not run. Splitting here
                        # separates "Python cannot launch fast enough" from "the stream is
                        # behind": CUDA events span stream idle time too, so their elapsed
                        # time alone cannot tell those apart.
                        launched_at = time.perf_counter()
                        # Hand over only settled memory, and time the GPU rather than the launch.
                        stream.synchronize()
                        if ev0 is not None:
                            gpu_s = ev0.elapsed_time(ev1) / 1000.0
                            launch_s = launched_at - started
                            sync_s = time.perf_counter() - launched_at
                    else:
                        actions = fn(batch)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the control thread
                error = exc
            finally:
                if self._sampler is not None:
                    self._sampler.set_active(False)
            prof = (wait_s, gpu_s, launch_s, sync_s) if self._profile else None
            self._results.put(
                (actions, step, generation, time.perf_counter() - started, error, prof)
            )

    def submit(self, fn, batch, step: int) -> None:
        self._inflight += 1
        self._requests.put((fn, batch, step, self._generation, time.perf_counter()))

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
        actions, step, generation, elapsed, error, prof = item
        self._inflight = max(0, self._inflight - 1)
        if error is not None:
            raise error
        if generation != self._generation:
            return None
        return actions, step, elapsed, prof

    def close(self) -> None:
        """Let the worker return before the interpreter tears the thread down."""
        if not self._thread.is_alive():
            return
        self._requests.put(None)
        # Generous enough for an in-flight forward to finish; it is a daemon either way.
        self._thread.join(timeout=5.0)

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
    __slots__ = ("ens", "step", "last_submit", "worker", "pending", "since_switch")

    def __init__(self, ens: ACTTemporalEnsembler):
        self.ens = ens
        self.step = 0
        # Far enough back that the first step is always due.
        self.last_submit = -(1 << 30)
        self.worker: _Worker | None = None
        self.pending: ACTTemporalEnsembler | None = None  # chunks landed mid-commit
        self.since_switch = 0


def install(
    coeff: float,
    every: int = 1,
    use_async: bool = True,
    align: bool = True,
    amp: str = "off",
    profile: bool = False,
    noise_scale: float = 1.0,
    samples: int = 1,
    worker_kind: str = "process",
    merge: str = "average",
    commit: int = 0,
    reduce: str = "mean",
    fade: int = 0,
) -> _Stats:
    """Replace SmolVLA's chunk queue with a temporal ensembler."""
    stats = _Stats()
    sampler = _Sampler() if profile else None
    stats.sampler = sampler
    original_reset = SmolVLAPolicy.reset

    def _state(self) -> _State:
        state = getattr(self, "_ens_state", None)
        if state is None:
            state = _State(ACTTemporalEnsembler(coeff, self.config.chunk_size))
            state.ens.reset()
            state.pending = ACTTemporalEnsembler(coeff, self.config.chunk_size)
            state.pending.reset()
            self._ens_state = state
        return state

    def _predict(self, batch, noise=None) -> torch.Tensor:
        return _predict_chunk(self, batch, noise, samples, noise_scale, reduce)

    def _make_worker(self):
        if worker_kind == "process":
            w = _ProcWorker(self, amp, samples, noise_scale, reduce)
            stats.proc_worker = w
            return w
        return _Worker(next(self.parameters()).device, amp, profile, sampler)

    def _trim(self, state: _State, landed):
        """Record a landed chunk and cut it to start at the current timestep (None if too late)."""
        actions, request_step, elapsed, prof = landed
        lag = state.step - request_step if align else 0
        stats.observe(elapsed, state.step - request_step, prof)

        # The ensembler is sized for chunk_size; a policy returning more would silently
        # misalign the weights, so keep only what it expects.
        chunk = self.config.chunk_size
        if actions.shape[1] > chunk:
            actions = actions[:, :chunk]
        if lag >= actions.shape[1]:
            stats.dropped += 1
            return None
        if lag > 0:
            # actions[:, j] is the prediction for timestep request_step + j; the buffer head
            # is timestep state.step. Drop the j < lag entries that are already in the past.
            actions = actions[:, lag:]
        return actions

    def _has(e: ACTTemporalEnsembler) -> bool:
        return e.ensembled_actions is not None and e.ensembled_actions.shape[1] > 0

    def _set(e: ACTTemporalEnsembler, actions: torch.Tensor) -> None:
        e.ensembled_actions = actions.clone()
        e.ensembled_actions_count = torch.ones(
            (actions.shape[1], 1), dtype=torch.long, device=actions.device
        )

    def _land(self, state: _State, landed) -> None:
        """commit mode: park a landed chunk in the pending buffer instead of executing it."""
        actions = _trim(self, state, landed)
        if actions is None:
            return
        if merge == "latest":
            if _has(state.pending):
                stats.superseded += 1
            _set(state.pending, actions)
        else:
            _merge(state.pending, actions)

    def _switch(state: _State) -> None:
        """commit mode: the pending buffer takes over, cross-fading in under `average`."""
        new = _crossfade(state.ens.ensembled_actions, state.pending.ensembled_actions, merge, fade)
        _set(state.ens, new)
        state.pending.ensembled_actions = None
        state.pending.ensembled_actions_count = None
        state.since_switch = 0

    def _fold(self, state: _State, landed, replace: bool = False) -> None:
        """Align a landed chunk to the current timestep and merge it."""
        actions = _trim(self, state, landed)
        if actions is None:
            return
        if merge == "latest" or replace:
            # No averaging: the newest chunk, already trimmed to the current timestep, takes
            # over outright. Its horizon is shorter than the old buffer's by `lag`, which is
            # fine -- the next chunk lands long before it runs out.
            state.ens.ensembled_actions = actions.clone()
            state.ens.ensembled_actions_count = torch.ones(
                (actions.shape[1], 1), dtype=torch.long, device=actions.device
            )
            return
        _merge(state.ens, actions)

    def reset(self):
        original_reset(self)
        state = _state(self)
        state.ens = ACTTemporalEnsembler(coeff, self.config.chunk_size)
        state.ens.reset()
        state.step = 0
        state.last_submit = -(1 << 30)
        state.pending = ACTTemporalEnsembler(coeff, self.config.chunk_size)
        state.pending.reset()
        state.since_switch = 0
        if state.worker is not None:
            state.worker.flush()
        elif use_async and worker_kind == "process" and getattr(self.config, "pretrained_path", None):
            # Start the process as early as possible -- reset() first runs from the policy's
            # constructor, long before the robot connects -- so its ~15 s model load overlaps
            # the robot's own start-up instead of freezing the first control step.
            state.worker = _make_worker(self)

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
                with _amp(amp, next(self.parameters()).device):
                    actions = _predict(self, batch, noise)  # (batch, chunk, action_dim)
                # Nothing was consumed while the loop blocked, so the chunk is still aligned.
                _fold(self, state, (actions, state.step, time.perf_counter() - started, None))
                state.last_submit = state.step
            state.step += 1
            return _pop(ens)

        if state.worker is None:
            state.worker = _make_worker(self)
        worker = state.worker
        if isinstance(worker, _ProcWorker):
            worker.wait_ready()

        # 1. Fold in whatever finished while the robot was being driven.
        landed = worker.poll()
        if landed is not None:
            if commit:
                _land(self, state, landed)  # parked: the robot is committed to its chunk
            else:
                _fold(self, state, landed)
        if commit and _has(state.pending) and (dry() or state.since_switch >= commit):
            _switch(state)

        # 2. Queue the next prediction. `dict(batch)` because _get_action_chunk mutates it.
        if not worker.busy and state.step - state.last_submit >= every:
            worker.submit(functools.partial(_predict, self), dict(batch), state.step)
            state.last_submit = state.step

        # 3. Only ever block when there is genuinely nothing to send -- the first step of an
        #    episode, or a stall long enough to drain a whole chunk.
        while dry():
            if commit and _has(state.pending):
                _switch(state)
                continue
            if not worker.busy:
                worker.submit(functools.partial(_predict, self), dict(batch), state.step)
                state.last_submit = state.step
            landed = worker.wait()
            if landed is not None:
                if commit:
                    _land(self, state, landed)
                else:
                    _fold(self, state, landed)

        state.step += 1
        state.since_switch += 1
        if commit and _has(state.pending):
            _pop(state.pending)  # keep the parked chunks aligned to the advancing timestep
        return _pop(ens)

    SmolVLAPolicy.reset = reset
    SmolVLAPolicy.select_action = select_action
    return stats


def install_base_rate_compensation(record_hz: float, floor: float = 0.7) -> None:
    """Scale the Mobile AI base velocity command so each control step covers the demo's distance.

    The policy's actions are indexed in *steps* of the recording run (~21.5 Hz on this robot,
    whatever the dataset's nominal fps says). The arms honour that -- one position target per step
    -- but the base velocity is held for however long the eval step actually lasts, so a loop at
    18.6 Hz (``--ensemble.samples=7``) drives every base segment 21.5/18.6 = 1.16x as far as the
    demonstration did while the arms follow the demonstration's path. Multiplying the command by
    ``measured_hz / record_hz`` keeps arm and base on the same per-step geometry.

    Only ever slows the base (factor capped at 1.0, floored at ``floor``); the period is an EMA of
    the interval between ``send_action`` calls; a gap over 1 s (episode save/reset) clears it, so the
    first command of each phase goes out unscaled. Applies to every ``send_action``, so the teleop
    reset phase is scaled too.
    """
    from lerobot_robot_trossen.mobileai import MobileAIRobot

    original = MobileAIRobot.send_action
    state = {"prev": None, "ema_dt": None, "calls": 0, "scale_sum": 0.0}
    period = 1.0 / record_hz

    @functools.wraps(original)
    def send_action(self, action):
        now = time.perf_counter()
        prev, state["prev"] = state["prev"], now
        if prev is not None and now - prev >= 1.0:
            # Episode save/reset pause: the next phase (teleop reset <-> policy) can run at a
            # different rate, so do not carry the old phase's period into it.
            state["ema_dt"] = None
        elif prev is not None:
            dt = now - prev
            state["ema_dt"] = dt if state["ema_dt"] is None else 0.9 * state["ema_dt"] + 0.1 * dt
        scale = 1.0
        if state["ema_dt"] is not None:
            scale = max(floor, min(1.0, period / state["ema_dt"]))
        if scale < 1.0 and ("x.vel" in action or "theta.vel" in action):
            action = dict(action)
            for k in ("x.vel", "theta.vel"):
                if k in action:
                    action[k] = float(action[k]) * scale
        state["calls"] += 1
        state["scale_sum"] += scale
        if state["calls"] % 300 == 0 and state["ema_dt"] is not None:
            logging.info(
                f"base rate compensation: 최근 루프 {1 / state['ema_dt']:.1f} Hz vs 녹화 {record_hz:g} Hz "
                f"-> 베이스 속도 x{scale:.3f} (평균 x{state['scale_sum'] / state['calls']:.3f})"
            )
        return original(self, action)

    MobileAIRobot.send_action = send_action
    logging.info(f"base rate compensation 켬: 녹화 주파수 {record_hz:g} Hz 기준으로 베이스 속도를 줄인다 (늘리지 않음)")


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
    profile = _pop_flag("--ensemble.profile", False)
    noise_scale = float(_pop_arg("--ensemble.noise_scale", "1.0"))
    if noise_scale < 0:
        raise ValueError(f"--ensemble.noise_scale 은 0 이상이어야 한다 (받은 값: {noise_scale})")
    merge = _pop_arg("--ensemble.merge", "average").strip().lower()
    if merge not in ("average", "latest"):
        raise ValueError(f"--ensemble.merge 는 average/latest 여야 한다 (받은 값: {merge!r})")
    commit = int(_pop_arg("--ensemble.commit", "0"))
    if commit < 0:
        raise ValueError(f"--ensemble.commit 은 0 이상이어야 한다 (받은 값: {commit})")
    fade = int(_pop_arg("--ensemble.fade", "0"))
    if fade < 0:
        raise ValueError(f"--ensemble.fade 는 0 이상이어야 한다 (받은 값: {fade})")
    worker_kind = _pop_arg("--ensemble.worker", "process").strip().lower()
    if worker_kind not in ("process", "thread"):
        raise ValueError(f"--ensemble.worker 는 process/thread 여야 한다 (받은 값: {worker_kind!r})")
    samples = int(_pop_arg("--ensemble.samples", "1"))
    if samples < 1:
        raise ValueError(f"--ensemble.samples 는 1 이상이어야 한다 (받은 값: {samples})")
    reduce = _pop_arg("--ensemble.reduce", "mean").strip().lower()
    if reduce not in ("mean", "medoid"):
        raise ValueError(f"--ensemble.reduce 는 mean/medoid 여야 한다 (받은 값: {reduce!r})")
    amp = _pop_arg("--ensemble.amp", "off").strip().lower()
    if amp not in ("off", "bf16", "fp16"):
        raise ValueError(f"--ensemble.amp 은 off/bf16/fp16 이어야 한다 (받은 값: {amp!r})")
    extra = _pop_arg("--ensemble.plugin", "")
    base_rate_hz = float(_pop_arg("--ensemble.base_rate_hz", "0"))
    init_logging()

    # main() 안에서도 호출되지만, 여기서 먼저 부르고 결과를 찍어 둔다 -- 플러그인이 안 잡힌 채
    # 파서가 돌면 "invalid choice: 'mobileai_robot'" 만 나와서 원인을 알기 어렵다.
    register_third_party_plugins()
    if extra:
        import importlib

        importlib.import_module(extra)
        logging.info(f"추가 플러그인 임포트: {extra}")
    _report_registered()
    if base_rate_hz > 0:
        install_base_rate_compensation(base_rate_hz)

    stats = install(
        coeff, every, use_async, align, amp, profile, noise_scale, samples, worker_kind, merge,
        commit, reduce, fade,
    )
    if commit:
        if not use_async:
            raise ValueError("--ensemble.commit 은 비동기 모드 전용이다 (--ensemble.async=false 와 같이 못 쓴다)")
        logging.info(
            f"commit={commit}: 워커는 쉬지 않고 추론하고, 로봇은 청크 하나를 {commit} 스텝 실행한 뒤 "
            + ("그 사이 도착한 청크들의 가중평균으로 크로스페이드해 갈아탄다 (TA, coeff 적용)"
               if merge != "latest" else "그때 도착해 있는 가장 새 청크로 통째로 갈아탄다 (평균 없음)")
            + ("" if every == 1 else f" -- 경고: every={every} 라 GPU 가 다시 쉰다. every 는 빼는 게 맞다")
        )
    if merge == "latest":
        logging.info("청크 병합: latest -- 시간축 평균(TA) 없음, 새 청크가 도착하면 통째로 교체 (coeff 무시)")
    if noise_scale != 1.0 or samples != 1:
        logging.info(
            f"샘플링 조정: 초기 노이즈 x{noise_scale:g}, 추론당 노이즈 샘플 {samples}개 "
            + ("평균" if reduce == "mean" else "중 medoid 하나 (평균 안 함)")
            + (" (결정적: 같은 관측이면 같은 청크)" if noise_scale == 0 else "")
        )
    cadence = "매 제어 스텝마다" if every == 1 else f"{every} 스텝마다"
    if use_async:
        logging.info(
            f"temporal ensembling 활성화 (coeff={coeff}, every={every}, async/{worker_kind}, "
            f"align={'on' if align else 'off'}, amp={amp}). 추론은 전용 스레드·CUDA 스트림에서 돌고 제어 "
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
