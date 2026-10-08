"""예측 청크 자체의 모양: 30스텝 안에서 팔 이동이 앞쪽에 몰리고 뒤는 평평한가(머물기 수렴). 실기 1123 프레임·시연 프레임, 4라운드 vs M1. 로봇 무접촉."""
import glob, os, numpy as np, torch, pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda"); ARM=[0,1,2,3,4,5,7,8,9,10,11,12]
def snap(m): return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--kiroaiseoul--{m}/snapshots/*/"))[0]
E15 = "1008_1128_chain_r4_t04_e15"
SETS = [("실기 1123 (e30)", "kiroaiseoul/eval_1008_1123_chain_r4_t04_e30", 0, [52+42, 52+252, 52+420])] + ([("실기 e15", f"kiroaiseoul/eval_{E15}", 0, [52+42, 52+252, 52+420])] if E15 else []) + [("시연 t04 ep0", "kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker", 0, [30, 90, 150]), ("시연 t04 ep60", "kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker", 60, [30, 90, 150])]
data=[]
for label, repo, e, frames in SETS:
    ds = LeRobotDataset(repo, episodes=[e]); fr=[f for f in frames if f < len(ds)]; data.append((label, ds.fps, fr, [ds[f] for f in fr]))
print("예측 청크 30스텝의 팔 이동을 6구간(5스텝씩)으로 나눈 L2 [rad] · 청크 첫 스텝 − 현재 state · p 끝값")
for name, m, env in [("4라운드 60K", "NAJY_act_all11_r4_27D_60k_s1000", True), ("M1", "NAJY_act_all11_hot_27D_120k_s1000", False)]:
    ck = snap(m); pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
    cams = [c for c in pol.config.input_features if c.startswith("observation.images")]
    print(f"\n== {name}")
    for label, fps, frames, items in data:
        for f, it in zip(frames, items):
            hot = np.zeros(11, np.float32); hot[3] = 1
            st = it["observation.state"].numpy()[:14].astype(np.float32)
            obs = {"observation.state": np.concatenate([st, np.zeros(2, np.float32), hot])}
            if env: obs["observation.environment_state"] = hot.copy()
            for c in cams: obs[c] = (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            with torch.inference_mode():
                a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
            segs = [np.linalg.norm(a[min(j+5,29),ARM]-a[j,ARM]) for j in range(0,30,5)]
            jump = np.linalg.norm(a[0,ARM]-st[ARM]); p = a[-1,16] if a.shape[1]>16 else float('nan')
            t = (f-52)/fps if "실기" in label else f/fps
            print(f"   {label:<14} t={t:5.1f}s: " + " ".join(f"{v:.3f}" for v in segs) + f" | 점프 {jump:.3f}" + (f" | p {p:.2f}" if p==p else ""))
    del pol; torch.cuda.empty_cache()
