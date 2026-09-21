"""Noise-sample ensembling: run K noise samples as one batch, average the K chunks.
Measures cost, and how much it shrinks chunk-to-chunk disagreement (the seam)."""
import time, torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
P = "/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model"
dev = torch.device("cuda:0")
pol = SmolVLAPolicy.from_pretrained(P).to(dev).eval()
pol.config.num_steps = 20
L = 20
torch.manual_seed(0)
obs = []
for _ in range(4):
    b = {OBS_STATE: torch.randn(1, 16, device=dev) * 0.5,
         OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, L), device=dev),
         OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, L, dtype=torch.bool, device=dev)}
    for k in pol.config.input_features:
        if "images" in k:
            b[k] = torch.rand(1, 3, 480, 640, device=dev)
    obs.append(b)
C, D = pol.config.chunk_size, pol.config.max_action_dim

def predict(b, K, scale=1.0):
    rep = {k: v.expand(K, *v.shape[1:]).contiguous() for k, v in b.items()}
    noise = torch.randn(K, C, D, device=dev) * scale
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pol._get_action_chunk(rep, noise)
    return out.float().mean(0, keepdim=True)

print("num_steps=20, bf16")
print(f"{'방식':<26} | {'추론':>7} | 같은 관측에서 독립 2회 예측의 차이(RMS) = 이음매 불일치")
with torch.inference_mode():
    for label, K, scale in (("K=1 (현재)", 1, 1.0), ("K=2 평균", 2, 1.0), ("K=4 평균", 4, 1.0),
                            ("K=8 평균", 8, 1.0), ("K=16 평균", 16, 1.0),
                            ("K=1, 노이즈 0.5배", 1, 0.5), ("K=1, 노이즈 0 (결정적)", 1, 0.0)):
        ts, rms = [], []
        for b in obs:
            for _ in range(3):
                t = time.perf_counter(); a1 = predict(b, K, scale); torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
                a2 = predict(b, K, scale)
                rms.append((a1 - a2).pow(2).mean().sqrt().item())
        ts = ts[2:]
        print(f"{label:<26} | {sum(ts)/len(ts)*1000:>4.0f} ms | {sum(rms)/len(rms):.4f}", flush=True)
