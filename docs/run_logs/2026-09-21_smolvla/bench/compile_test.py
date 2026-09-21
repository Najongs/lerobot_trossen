"""Does torch.compile(mode="reduce-overhead") survive the ensemble worker's
private CUDA stream on a worker thread, and how long does the first call cost?"""
import threading, time, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

P = "/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model"
dev = torch.device("cuda")

def build(compile_model):
    pol = SmolVLAPolicy.from_pretrained(P, compile_model=compile_model,
                                        compile_mode="reduce-overhead").to(dev).eval()
    return pol

L = 20
def make_batch(pol):
    b = {OBS_STATE: torch.zeros(1, 16, device=dev),
         OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, L), device=dev),
         OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, L, dtype=torch.bool, device=dev)}
    for k in pol.config.input_features:
        if "images" in k:
            b[k] = torch.rand(1, 3, 480, 640, device=dev)
    return b

def run(compile_model, n=8):
    pol = build(compile_model); batch = make_batch(pol)
    out, err = [], None
    def work():
        nonlocal err
        try:
            stream = torch.cuda.Stream(device=dev)          # exactly what _Worker does
            with torch.inference_mode():
                for _ in range(n):
                    t = time.perf_counter()
                    with torch.cuda.stream(stream):
                        pol._get_action_chunk(dict(batch))
                    stream.synchronize()
                    out.append(time.perf_counter() - t)
        except BaseException as e:
            err = e
    w = threading.Thread(target=work); w.start(); w.join()
    tag = "compile=on " if compile_model else "compile=off"
    if err is not None:
        print(f"{tag}: FAILED -> {type(err).__name__}: {err}", flush=True)
        return
    steady = sum(out[2:]) / len(out[2:]) * 1000
    print(f"{tag}: 1st(컴파일) {out[0]:8.1f} s | 2nd {out[1]*1000:7.0f} ms | 안정 {steady:6.0f} ms"
          .replace(f"{out[0]:8.1f} s", f"{out[0]:8.1f} s"), flush=True)

print("워커 스레드 + 전용 CUDA 스트림에서 측정", flush=True)
run(False)
run(True)
