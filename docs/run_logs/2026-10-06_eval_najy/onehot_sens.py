"""원핫 민감도 -- 같은 관측(카메라+팔 state)에 원핫만 바꿔 넣고 M1 의 action 청크를 비교한다.

state 27D = [팔 14, 베이스 0 0, 원핫 11]  (task_onehot_patch.TaskOneHotStep 과 같은 배치)
로봇 무접촉. 사용: python onehot_sens.py <체크포인트 경로> <데이터셋 이름> [stride]
"""
import sys, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference

CKPT, REPO = sys.argv[1], sys.argv[2]
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 21
EPS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
dev = torch.device("cuda")
policy = ACTPolicy.from_pretrained(CKPT).to(dev).eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=CKPT)
ds = LeRobotDataset(REPO, episodes=EPS)
cams = [k for k in policy.config.input_features if k.startswith("observation.images")]
J = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # 팔 12관절 (그리퍼 6·13 제외)

VARIANTS = {"t04(4/11)": 3, "t05(5/11)": 4, "zero": None, "t01": 0, "t07": 6, "t11": 10}


def chunk(item, hot_idx):
    st = item["observation.state"].numpy()[:14].astype(np.float32)
    hot = np.zeros(11, np.float32)
    if hot_idx is not None:
        hot[hot_idx] = 1
    obs = {"observation.state": np.concatenate([st, np.zeros(2, np.float32), hot])}
    for c in cams:
        obs[c] = (item[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    with torch.inference_mode():
        b = pre(prepare_observation_for_inference(obs, dev, None, None))
        a = post(policy.predict_action_chunk(b))
    return a[0].float().cpu().numpy(), st  # (chunk, 16)


idx = list(range(0, len(ds), STRIDE))
diffs = {k: [] for k in VARIANTS}
scale = []
for i in idx:
    item = ds[i]
    out = {k: chunk(item, v)[0] for k, v in VARIANTS.items()}
    st = item["observation.state"].numpy()
    ref = out["t05(5/11)"]
    # 청크가 그리는 움직임 크기: 청크 끝 - 현재 state (팔 12관절 L2)
    scale.append(np.linalg.norm(ref[-1, J] - st[J]))
    for k in VARIANTS:
        diffs[k].append(np.linalg.norm(out[k][:, J] - ref[:, J], axis=1).mean())

print(f"{REPO}: {len(idx)} 프레임 (stride {STRIDE}) | 기준 = 원핫 5/11")
print(f"  청크 움직임 크기 |a_5[-1]-state| 중앙 {np.median(scale):.3f} rad")
for k in VARIANTS:
    d = np.array(diffs[k])
    print(f"  {k:10s} vs 5/11: 청크 평균 L2 차 중앙 {np.median(d):.4f}  p90 {np.percentile(d, 90):.4f}  최대 {d.max():.4f}")

# --- 예측 청크가 어느 시연 쪽으로 가나 / 정답 action 과의 오차 (학습 데이터일 때) ---
import glob, os, pandas as pd
H = os.path.expanduser("~/.cache/huggingface/lerobot/kiroaiseoul/")
def load_states(name):
    fs = sorted(glob.glob(H + name + "/data/*/*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["observation.state"]) for f in fs])
    return np.stack(d["observation.state"].values)[::3][:, J]
T4 = torch.tensor(load_states("task04_pour_liquid_from_tubes_to_beaker"), device=dev)
T5 = torch.tensor(load_states("task05_tube_disposal"), device=dev)
def near(x, T):
    return torch.cdist(torch.tensor(x, device=dev, dtype=T.dtype), T).min(1).values.cpu().numpy()

gt = None
fs = sorted(glob.glob(H + REPO.split("/")[1] + "/data/*/*.parquet"))
d = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action"]) for f in fs])
gt = {(e, f): a for e, f, a in zip(d["episode_index"], d["frame_index"], d["action"])}
CH = policy.config.chunk_size
res = {k: {"t04": [], "t05": [], "gt": []} for k in ["t04(4/11)", "t05(5/11)", "zero"]}
for i in idx:
    item = ds[i]; e = int(item["episode_index"]); f0 = int(item["frame_index"])
    g = [gt.get((e, f0 + j)) for j in range(CH)]
    for k in res:
        a = chunk(item, VARIANTS[k])[0][:, J]
        res[k]["t04"].append(np.median(near(a, T4))); res[k]["t05"].append(np.median(near(a, T5)))
        if all(x is not None for x in g) and "eval_" not in REPO:
            res[k]["gt"].append(np.linalg.norm(a - np.stack(g)[:, J], axis=1).mean())
print("  예측 청크 자세의 시연 최근접 (중앙)  |  정답 action 청크와 평균 L2 오차 (학습 데이터만)")
for k, r in res.items():
    gte = f"{np.median(r['gt']):.4f} (n={len(r['gt'])})" if r["gt"] else "-"
    win = np.mean(np.array(r["t04"]) < np.array(r["t05"])) * 100
    print(f"  {k:10s} t04 {np.median(r['t04']):.3f}  t05 {np.median(r['t05']):.3f}  t04쪽 {win:3.0f}%  |  {gte}")
