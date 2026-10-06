"""같은 관측에 M1(원핫 1/11)·M3(원핫 1/4)를 넣고 청크의 베이스 명령(x.vel, theta.vel)을 본다. 로봇 무접촉.
사용: python base_pred.py <데이터셋> <stride> [episodes]  (모델 경로는 HF 캐시)"""
import sys, glob, os, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda")
def snap(m): return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--kiroaiseoul--{m}/snapshots/*/"))[0]
MODELS = {"M1 1/11": (snap("NAJY_act_all11_hot_27D_120k_s1000"), 11, 0), "M3 1/4": (snap("NAJY_act_move4_hot_20D_60k_s1000"), 4, 0)}
REPO, STRIDE = sys.argv[1], int(sys.argv[2]); EPS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
ds = LeRobotDataset(REPO, episodes=EPS)
idx = list(range(0, min(len(ds), int(os.environ.get("MAXF", 10**9))), STRIDE))
gt = np.stack([ds.hf_dataset[i]["action"].numpy() if hasattr(ds.hf_dataset[i]["action"], "numpy") else np.array(ds.hf_dataset[i]["action"]) for i in idx])
print(f"{REPO}: {len(idx)} 프레임 | 데이터의 action x/θ 평균 {gt[:,14].mean():+.3f} / {gt[:,15].mean():+.3f}  (|x|>0.05 프레임 {np.mean(np.abs(gt[:,14])>0.05)*100:.0f}%)")
for name, (ck, k, h) in MODELS.items():
    pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
    cams = [c for c in pol.config.input_features if c.startswith("observation.images")]
    xs, ths = [], []
    for i in idx:
        it = ds[i]; hot = np.zeros(k, np.float32); hot[h] = 1
        obs = {"observation.state": np.concatenate([it["observation.state"].numpy()[:14].astype(np.float32), np.zeros(2, np.float32), hot])}
        for c in cams: obs[c] = (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        with torch.inference_mode():
            a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
        xs.append(a[:, 14].mean()); ths.append(a[:, 15].mean())
    xs, ths = np.array(xs), np.array(ths)
    print(f"  {name}: 청크 평균 x.vel {xs.mean():+.3f} (최대 {xs.max():+.3f}, >0.05 {np.mean(xs>0.05)*100:.0f}%) | theta.vel {ths.mean():+.3f} (|θ| 최대 {np.abs(ths).max():.3f})")
    del pol; torch.cuda.empty_cache()
