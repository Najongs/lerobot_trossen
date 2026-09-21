"""Separate GIL contention from GPU/PCIe contention at the load level that reproduces
the 8x. Same 4 saturated threads; vary only what they touch."""
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

# mode: "cpu" = numpy only, "h2d" = upload to GPU, no readback, "full" = upload + .cpu()
def saturated(stop, mode):
    while not stop.is_set():
        for f in frames:
            if mode == "cpu":
                _ = f.astype(np.float32).mean()
                _ = np.ascontiguousarray(f.transpose(2, 0, 1))
            else:
                y = torch.from_numpy(f).to(dev, non_blocking=True).permute(2, 0, 1).float().div_(255)
                if mode == "full":
                    y.cpu()

def measure(label, n_ctl, mode, switch, n=6):
    sys.setswitchinterval(switch)
    stop = threading.Event()
    ts = [threading.Thread(target=saturated, args=(stop, mode), daemon=True) for _ in range(n_ctl)]
    for t in ts: t.start()
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
    stop.set()
    for t in ts: t.join()
    sys.setswitchinterval(0.005)
    warm = out[1:]
    print(f"{label:<50} {sum(warm)/len(warm)*1000:7.0f} ms", flush=True)

measure("경쟁 없음", 0, "cpu", 0.005)
measure("4x 포화, CPU(numpy)만          switch 5ms", 4, "cpu", 0.005)
measure("4x 포화, CPU(numpy)만          switch 0.5ms", 4, "cpu", 0.0005)
measure("4x 포화, GPU 업로드만 (.cpu 없음)", 4, "h2d", 0.005)
measure("4x 포화, GPU 업로드+.cpu() 왕복", 4, "full", 0.005)
