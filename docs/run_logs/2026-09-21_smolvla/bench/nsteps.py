import time, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
P = "/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model"
dev = torch.device("cuda:0")
pol = SmolVLAPolicy.from_pretrained(P).to(dev).eval()
L = 20
batch = {OBS_STATE: torch.zeros(1, 16, device=dev),
         OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, L), device=dev),
         OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, L, dtype=torch.bool, device=dev)}
for k in pol.config.input_features:
    if "images" in k:
        batch[k] = torch.rand(1, 3, 480, 640, device=dev)
noise = torch.randn(1, pol.config.chunk_size, pol.config.max_action_dim, device=dev)
ref = None
print(f"{'num_steps':>9} | {'fp32':>8} | {'bf16':>8} | 같은 노이즈에서 num_steps=100 대비 액션 차이(RMS)", flush=True)
with torch.inference_mode():
    pol.config.num_steps = 100
    ref = pol._get_action_chunk(dict(batch), noise.clone()).float()
    for n in (5, 10, 15, 20, 30, 50):
        pol.config.num_steps = n
        res = {}
        for label, ctx in (("fp32", None), ("bf16", torch.bfloat16)):
            ts = []
            for i in range(6):
                t = time.perf_counter()
                if ctx is None:
                    out = pol._get_action_chunk(dict(batch), noise.clone())
                else:
                    with torch.autocast("cuda", dtype=ctx):
                        out = pol._get_action_chunk(dict(batch), noise.clone())
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            res[label] = sum(ts[2:]) / len(ts[2:]) * 1000
            if ctx is None:
                rms = (out.float() - ref).pow(2).mean().sqrt().item()
        print(f"{n:>9} | {res['fp32']:>5.0f} ms | {res['bf16']:>5.0f} ms | {rms:.4f}", flush=True)
