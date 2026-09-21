"""Reproduce the async-worker starvation offline: inference on a worker thread with a
private CUDA stream, while a synthetic 'control thread' does what record_loop does at 20 Hz.

Then test the two candidate fixes:
  (A) GIL convoy      -> sys.setswitchinterval
  (B) stream contention -> control thread doing GPU work vs CPU-only work
"""
import sys, threading, time
import numpy as np
import torch
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

def control_thread(stop, gpu_work):
    """~20 Hz, doing per-step work shaped like record_loop's."""
    while not stop.is_set():
        t0 = time.perf_counter()
        for f in frames:
            x = torch.from_numpy(f)
            if gpu_work:                       # the LEROBOT_FAST_OBS path
                x = x.to(dev, non_blocking=True).permute(2, 0, 1).float().div_(255)
                x.cpu()                        # predict_action ends in a .cpu()
            else:
                _ = f.astype(np.float32) / 255.0
        dt = time.perf_counter() - t0
        time.sleep(max(0.0, 0.05 - dt))        # target 20 Hz

def measure(label, n_control, gpu_work, switch, n=6):
    sys.setswitchinterval(switch)
    stop = threading.Event()
    ctl = [threading.Thread(target=control_thread, args=(stop, gpu_work), daemon=True)
           for _ in range(n_control)]
    for t in ctl: t.start()
    out = []
    def work():
        stream = torch.cuda.Stream(device=dev)
        with torch.inference_mode():
            for _ in range(n):
                t = time.perf_counter()
                with torch.cuda.stream(stream):
                    pol._get_action_chunk(dict(batch))
                stream.synchronize()
                out.append(time.perf_counter() - t)
    w = threading.Thread(target=work); w.start(); w.join()
    stop.set()
    for t in ctl: t.join()
    sys.setswitchinterval(0.005)
    warm = out[1:]
    print(f"{label:<46} {sum(warm)/len(warm)*1000:7.0f} ms", flush=True)

measure("경쟁 없음", 0, False, 0.005)
measure("제어스레드 1개 (CPU만), switch 5ms", 1, False, 0.005)
measure("제어스레드 1개 (CPU+GPU), switch 5ms  <- 실제", 1, True, 0.005)
measure("제어스레드 1개 (CPU+GPU), switch 0.5ms", 1, True, 0.0005)
measure("제어스레드 1개 (CPU+GPU), switch 0.05ms", 1, True, 0.00005)
