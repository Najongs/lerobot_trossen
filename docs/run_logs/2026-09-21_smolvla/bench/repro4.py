"""Realistic-load reproduction: mirror what record_loop actually does per control step,
at the rate it actually achieves (~20 Hz), and see whether that alone reaches ~1162 ms.

Per step, matching lerobot_record.record_loop + fast_obs_patch + record_ensemble:
  - 3x (uint8 480x640x3 -> GPU -> permute -> float -> /255)      [fast_obs]
  - small normalisation ops on the default stream                 [preprocessor]
  - ensemble bookkeeping on the default stream                    [_merge/_pop]
  - one small D2H readback                                        [predict_action's .cpu()]
"""
import sys, threading, time
import numpy as np, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

P = "/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model"
dev = torch.device("cuda")
pol = SmolVLAPolicy.from_pretrained(P).to(dev).eval()
L = 20
batch = {OBS_STATE: torch.zeros(1, 16, device=dev),
         OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, L), device=dev),
         OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, L, dtype=torch.bool, device=dev)}
for k in pol.config.input_features:
    if "images" in k:
        batch[k] = torch.rand(1, 3, 480, 640, device=dev)
frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]

steps = {"n": 0}

def control_loop(stop, hz):
    """One iteration == one record_loop step."""
    period = (1.0 / hz) if hz else 0.0
    ens = torch.zeros(1, 50, 16, device=dev)        # the ensemble buffer
    while not stop.is_set():
        t0 = time.perf_counter()
        imgs = []
        for f in frames:                             # fast_obs: 3x H2D + convert
            imgs.append(torch.from_numpy(f).to(dev, non_blocking=True)
                        .permute(2, 0, 1).float().div_(255))
        for im in imgs:                              # preprocessor: normalise
            im.sub_(0.5).div_(0.5)
        w = torch.rand(1, 50, 1, device=dev)         # _merge: weighted average
        ens = (ens * w + torch.rand_like(ens) * (1 - w))
        action = ens[:, 0]                           # _pop
        action.cpu()                                 # predict_action's readback
        steps["n"] += 1
        if period:
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

def measure(label, hz, n=6):
    stop = threading.Event()
    steps["n"] = 0
    ctl = threading.Thread(target=control_loop, args=(stop, hz), daemon=True)
    ctl.start()
    t_start = time.perf_counter()
    out = []
    def work():
        s = torch.cuda.Stream(device=dev)
        with torch.inference_mode():
            for _ in range(n):
                t0 = time.perf_counter()
                with torch.cuda.stream(s):
                    pol._get_action_chunk(dict(batch))
                s.synchronize()
                out.append(time.perf_counter() - t0)
    w = threading.Thread(target=work); w.start(); w.join()
    elapsed = time.perf_counter() - t_start
    stop.set(); ctl.join()
    warm = out[1:]
    rate = steps["n"] / elapsed
    print(f"{label:<44} {sum(warm)/len(warm)*1000:7.0f} ms   (제어루프 실측 {rate:4.1f} Hz)", flush=True)

measure("경쟁 없음", None if False else 0) if False else None
print(f"{'조건':<44} {'추론':>10}", flush=True)
measure("제어루프 없음", -1) if False else None

# baseline with no control thread at all
def baseline(n=6):
    out = []
    def work():
        s = torch.cuda.Stream(device=dev)
        with torch.inference_mode():
            for _ in range(n):
                t0 = time.perf_counter()
                with torch.cuda.stream(s):
                    pol._get_action_chunk(dict(batch))
                s.synchronize()
                out.append(time.perf_counter() - t0)
    w = threading.Thread(target=work); w.start(); w.join()
    warm = out[1:]
    print(f"{'제어루프 없음':<44} {sum(warm)/len(warm)*1000:7.0f} ms", flush=True)

baseline()
measure("제어루프 20 Hz (실제 재현)", 20)
measure("제어루프 30 Hz", 30)
measure("제어루프 무제한", 0)
