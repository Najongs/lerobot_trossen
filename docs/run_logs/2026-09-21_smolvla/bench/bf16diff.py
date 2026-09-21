import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
P = "/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model"
dev = torch.device("cuda:0")
pol = SmolVLAPolicy.from_pretrained(P).to(dev).eval()
L = 20
torch.manual_seed(0)
def mk():
    b = {OBS_STATE: torch.randn(1, 16, device=dev) * 0.5,
         OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, L), device=dev),
         OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, L, dtype=torch.bool, device=dev)}
    for k in pol.config.input_features:
        if "images" in k:
            b[k] = torch.rand(1, 3, 480, 640, device=dev)
    return b
rows = []
with torch.inference_mode():
    for trial in range(5):
        batch = mk()
        noise = torch.randn(1, pol.config.chunk_size, pol.config.max_action_dim, device=dev)
        noise2 = torch.randn_like(noise)
        ref = pol._get_action_chunk(dict(batch), noise.clone()).float()
        other_noise = pol._get_action_chunk(dict(batch), noise2.clone()).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf = pol._get_action_chunk(dict(batch), noise.clone()).float()
        with torch.autocast("cuda", dtype=torch.float16):
            fp = pol._get_action_chunk(dict(batch), noise.clone()).float()
        rms = lambda a: (a - ref).pow(2).mean().sqrt().item()
        rows.append((rms(bf), (bf - ref).abs().max().item(), rms(fp), bool(torch.isnan(fp).any()), rms(other_noise)))
print("trial | bf16 RMS | bf16 max | fp16 RMS | fp16 NaN | (비교) 노이즈만 바꾼 fp32 RMS")
for i, r in enumerate(rows):
    print(f"{i:>5} | {r[0]:.4f}   | {r[1]:.4f}   | {r[2]:.4f}   | {r[3]!s:<8} | {r[4]:.4f}")
