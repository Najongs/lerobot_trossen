"""카메라 vs 팔 state -- 어느 쪽이 단계를 정하나. 원핫 5/11 고정, 이미지와 state 를 두 출처에서 섞어 넣는다.
A = C-2 실기 프레임(task04 처럼 행동), B = task05 학습 프레임. 예측 청크 자세가 task04/task05 시연 중 어디에 가까운지.
로봇 무접촉. 사용: python swap.py <체크포인트>"""
import sys, glob, os, numpy as np, torch, pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
CKPT = sys.argv[1]; dev = torch.device("cuda")
policy = ACTPolicy.from_pretrained(CKPT).to(dev).eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=CKPT)
cams = [k for k in policy.config.input_features if k.startswith("observation.images")]
J = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
H = os.path.expanduser("~/.cache/huggingface/lerobot/kiroaiseoul/")
def states(name):
    fs = sorted(glob.glob(H + name + "/data/*/*.parquet"))
    return np.stack(pd.concat([pd.read_parquet(f, columns=["observation.state"]) for f in fs])["observation.state"].values)[::3][:, J]
T4 = torch.tensor(states("task04_pour_liquid_from_tubes_to_beaker"), device=dev)
T5 = torch.tensor(states("task05_tube_disposal"), device=dev)
def side(a):
    x = torch.tensor(a, device=dev, dtype=T4.dtype)
    return np.median(torch.cdist(x, T4).min(1).values.cpu().numpy()), np.median(torch.cdist(x, T5).min(1).values.cpu().numpy())
def run(img_item, st14):
    hot = np.zeros(11, np.float32); hot[4] = 1
    obs = {"observation.state": np.concatenate([st14.astype(np.float32), np.zeros(2, np.float32), hot])}
    for c in cams: obs[c] = (img_item[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    with torch.inference_mode():
        return post(policy.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()[:, J]
A = LeRobotDataset("kiroaiseoul/eval_najy_1006_1054_m1_t05_e30")
B = LeRobotDataset("kiroaiseoul/task05_tube_disposal", episodes=[0, 10, 20, 30, 40])
ia = list(range(0, len(A), 21))[:60]; ib = list(range(0, len(B), max(1, len(B) // len(ia))))[:len(ia)]
n = min(len(ia), len(ib)); ia, ib = ia[:n], ib[:n]

def run_hot(img_item, st14, h):
    hot = np.zeros(11, np.float32); hot[h] = 1
    obs = {"observation.state": np.concatenate([st14.astype(np.float32), np.zeros(2, np.float32), hot])}
    for c in cams: obs[c] = (img_item[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    with torch.inference_mode():
        return post(policy.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()[:, J]
L = lambda x, y: np.linalg.norm(x - y, axis=1).mean()
eff = {"원핫 4↔5 (이미지·state 고정)": [], "이미지 A↔B (state 고정)": [], "state A↔B (이미지 고정, 상대 움직임)": [], "청크 상대 움직임 크기 |a−s|": []}
for a_i, b_i in zip(ia, ib):
    a, b = A[a_i], B[b_i]
    sa, sb = a["observation.state"].numpy()[:14], b["observation.state"].numpy()[:14]
    ja, jb = sa[J], sb[J]
    for img, st, js in ((a, sa, ja), (b, sb, jb)):
        eff["원핫 4↔5 (이미지·state 고정)"].append(L(run_hot(img, st, 3), run_hot(img, st, 4)))
        eff["청크 상대 움직임 크기 |a−s|"].append(L(run_hot(img, st, 4), js))
    for st in (sa, sb):
        eff["이미지 A↔B (state 고정)"].append(L(run_hot(a, st, 4), run_hot(b, st, 4)))
    for img in (a, b):
        eff["state A↔B (이미지 고정, 상대 움직임)"].append(L(run_hot(img, sa, 4) - ja, run_hot(img, sb, 4) - jb))
print(f"A = C-2 실기 프레임, B = task05 학습 프레임, 쌍 {n}개, 기본 원핫 5/11 -- 팔 12관절 청크 평균 L2 (rad)")
for k, v in eff.items():
    print(f"  {k:36s} 중앙 {np.median(v):.3f}  p90 {np.percentile(v,90):.3f}")
