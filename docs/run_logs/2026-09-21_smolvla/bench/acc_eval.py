"""Open-loop accuracy: feed real demonstration frames, compare the predicted chunk with what the
demonstrator actually did next. Real units (rad / m / m/s), after the checkpoint's own postprocessor."""
import json, random, sys, time
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

P = "/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model"
ROOT = "/home/trossen-ai/.cache/huggingface/lerobot/kiroaiseoul/task02_pickup_tubes"
dev = torch.device("cuda:0")
info = json.load(open(f"{ROOT}/meta/info.json"))
names = info["features"]["action"]["names"]; names = names.get("motors", names) if isinstance(names, dict) else names
print("action dims:", names, flush=True)

EPISODES = [3, 27, 51, 77, 101, 125]
ds = LeRobotDataset("kiroaiseoul/task02_pickup_tubes", root=ROOT, episodes=EPISODES,
                    delta_timestamps={"action": [i / info["fps"] for i in range(50)]})
policy = SmolVLAPolicy.from_pretrained(P).to(dev).eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=P)

# frames: 6 phases of each episode
ep_idx = torch.tensor([int(x) for x in ds.hf_dataset["episode_index"]])
frames = []
for e in EPISODES:
    idx = (ep_idx == e).nonzero().flatten().tolist()
    frames += [idx[int(len(idx) * f)] for f in (0.02, 0.2, 0.4, 0.6, 0.8, 0.93)]
items = []
for i in frames:
    it = ds[i]
    obs = {k: v.unsqueeze(0) for k, v in it.items() if k.startswith("observation.")}
    obs["task"] = it["task"]
    batch = pre(obs)
    items.append((batch, it["action"].float(), it["action_is_pad"], it["observation.state"].float()))
print(f"frames: {len(items)}  task: {ds[frames[0]]['task']!r}", flush=True)

C, D = policy.config.chunk_size, policy.config.max_action_dim
def predict(batch, steps, ns, K):
    policy.config.num_steps = steps
    b = {k: (v.expand(K, *v.shape[1:]) if torch.is_tensor(v) and v.shape[0] == 1 else v) for k, v in batch.items()}
    noise = torch.randn(K, C, D, device=dev) * ns
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = policy._get_action_chunk(dict(b), noise).float().mean(0, keepdim=True)
    return post(out).squeeze(0).cpu()            # (50, 16) real units

grip = [i for i, n in enumerate(names) if "grip" in n.lower() or "carriage" in n.lower()]
base = [i for i, n in enumerate(names) if "vel" in n.lower()]
arm = [i for i in range(len(names)) if i not in grip and i not in base]
print("arm", arm, "| gripper", grip, "| base", base, flush=True)

def score(pred_fn, reps):
    se = {"arm": [], "grip": [], "base": [], "arm_0_9": [], "arm_30_49": []}
    for batch, gt, pad, state in items:
        for _ in range(reps):
            pr = pred_fn(batch, state)
            ok = ~pad
            d2 = (pr - gt) ** 2
            se["arm"].append(d2[ok][:, arm].mean().item()); se["grip"].append(d2[ok][:, grip].mean().item())
            se["base"].append(d2[ok][:, base].mean().item())
            se["arm_0_9"].append(d2[:10][ok[:10]][:, arm].mean().item())
            m = ok.clone(); m[:30] = False
            if m.any(): se["arm_30_49"].append(d2[m][:, arm].mean().item())
    return {k: (sum(v) / len(v)) ** 0.5 for k, v in se.items()}

rows = [("기준: 현재 자세 유지(정책 없음)", lambda b, s: s.unsqueeze(0).expand(50, -1).clone(), 1)]
for label, st, ns, K in (("stock  10스텝 ns1.0 K1", 10, 1.0, 1), ("사용중 10스텝 ns0.9 K7", 10, 0.9, 7),
                         ("10스텝 ns0.5 K4", 10, 0.5, 4), ("10스텝 ns0.5 K1", 10, 0.5, 1), ("10스텝 ns0 K1(결정적)", 10, 0.0, 1),
                         ("25스텝 ns1.0 K4", 25, 1.0, 4), ("50스텝 ns1.0 K1", 50, 1.0, 1), ("50스텝 ns0.5 K4", 50, 0.5, 4)):
    rows.append((label, (lambda b, s, st=st, ns=ns, K=K: predict(b, st, ns, K)), 3))
print(f"\n{'설정':<32} {'팔 RMSE(rad)':>12} {'  앞 0~9':>9} {'뒤 30~49':>9} {'그리퍼(m)':>10} {'base':>8}")
for label, fn, reps in rows:
    r = score(fn, reps)
    print(f"{label:<32} {r['arm']:>12.4f} {r['arm_0_9']:>9.4f} {r['arm_30_49']:>9.4f} {r['grip']:>10.5f} {r['base']:>8.4f}", flush=True)
