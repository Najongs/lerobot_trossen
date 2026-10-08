"""4라운드 모델에 t01 실기 프레임·시연 프레임을 넣고 원핫(+ENV 토큰)만 바꿔 가며 청크의 베이스 명령·진행도를 본다. 로봇 무접촉.
입력 배치: state 27D = 팔 14 + 0 0 + 원핫 11, observation.environment_state = 같은 원핫(패치와 동일)."""
import glob, os, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda")
ck = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--kiroaiseoul--NAJY_act_all11_r4_27D_60k_s1000/snapshots/*/"))[0]
pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
cams = [c for c in pol.config.input_features if c.startswith("observation.images")]
HOTS = [("1/11 (맞음)", 0), ("6/11", 5), ("10/11", 9), ("3/11", 2), ("0벡터", None)]
SETS = [("실기 1047 (뒤로 뺀 자리)", "kiroaiseoul/eval_1008_1047_chain_r4_t01_e30", 0, [52 + 21 * k for k in (0, 2, 4, 6, 8)]),
        ("실기 1046 (선반 가까이)", "kiroaiseoul/eval_1008_1046_chain_r4_t01_e30", 0, [52 + 21 * k for k in (0, 2, 4, 6, 8, 10)]),
        ("시연 task01 ep0", "kiroaiseoul/task01_move_to_tube_rack", 0, [0, 15, 30, 60, 90, 120]),
        ("시연 task01 ep50", "kiroaiseoul/task01_move_to_tube_rack", 50, [0, 15, 30, 60, 90, 120])]
print("청크(30) 평균: x.vel / theta.vel [m/s, rad/s] · p(17번째 칸) 끝값 — 열 = 프레임 시각(s, 표기 fps 기준)")
for label, repo, e, frames in SETS:
    ds = LeRobotDataset(repo, episodes=[e]); fps = ds.fps; frames = [f for f in frames if f < len(ds)]
    items = [ds[f] for f in frames]
    print(f"\n== {label} | t: " + " ".join(f"{(f - (52 if 'eval' in repo else 0)) / fps:5.1f}" for f in frames))
    for name, h in HOTS:
        out = []
        for it in items:
            hot = np.zeros(11, np.float32)
            if h is not None: hot[h] = 1
            obs = {"observation.state": np.concatenate([it["observation.state"].numpy()[:14].astype(np.float32), np.zeros(2, np.float32), hot]),
                   "observation.environment_state": hot.copy()}
            for c in cams: obs[c] = (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            with torch.inference_mode():
                a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
            out.append((a[:, 14].mean(), a[:, 15].mean(), a[-1, 16]))
        print(f"  {name:<11}: " + " ".join(f"{x:+.2f}/{t:+.2f} p{p:.2f}" for x, t, p in out))
