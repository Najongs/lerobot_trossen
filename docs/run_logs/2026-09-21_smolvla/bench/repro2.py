"""Second attempt: saturate the control thread the way record_loop actually does
(~50 ms of work per 50 ms period, no idle), plus 3 camera-like threads."""
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

def saturated(stop, gpu):
    """No sleep: this is what a loop with ~50 ms of work per 50 ms period looks like."""
    while not stop.is_set():
        for f in frames:
            x = torch.from_numpy(f)
            if gpu:
                y = x.to(dev, non_blocking=True).permute(2, 0, 1).float().div_(255)
                y.cpu()
            _ = f.astype(np.float32).mean()
            _ = np.ascontiguousarray(f.transpose(2, 0, 1))

def measure(label, n_ctl, gpu, switch, n=6):
    sys.setswitchinterval(switch)
    stop = threading.Event()
    ts = [threading.Thread(target=saturated, args=(stop, gpu), daemon=True) for _ in range(n_ctl)]
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
    print(f"{label:<52} {sum(warm)/len(warm)*1000:7.0f} ms", flush=True)

measure("경쟁 없음", 0, False, 0.005)
measure("포화 스레드 1개 (CPU+GPU), switch 5ms", 1, True, 0.005)
measure("포화 스레드 4개 (CPU+GPU), switch 5ms  <- 실제 근사", 4, True, 0.005)
measure("포화 스레드 4개, switch 0.5ms", 4, True, 0.0005)
measure("포화 스레드 4개, switch 0.05ms", 4, True, 0.00005)
