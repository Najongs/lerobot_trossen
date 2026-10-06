"""task03 실기(1350) 프레임에 M1·M3 를 다시 넣어 청크의 회전 명령(theta.vel 평균)을 본다 — 원핫만 바꿔 가며. 로봇 무접촉.
사용: python t03_stall_pred.py   (모델·데이터는 HF 캐시, base_pred.py 와 같은 입력 배치: 팔 14 + 0 0 + 원핫)"""
import glob, os, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda")
def snap(m): return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--kiroaiseoul--{m}/snapshots/*/"))[0]
M1, M3 = snap("NAJY_act_all11_hot_27D_120k_s1000"), snap("NAJY_act_move4_hot_20D_60k_s1000")
# (이름, 체크포인트, 원핫 폭, 켜는 칸(0-based) 또는 None=0벡터)
CFG = [("M1 3/11", M1, 11, 2), ("M1 2/11", M1, 11, 1), ("M1 4/11", M1, 11, 3), ("M1 1/11", M1, 11, 0), ("M1 0벡터", M1, 11, None), ("M3 2/4", M3, 4, 1), ("M3 1/4", M3, 4, 0)]
EVAL = "kiroaiseoul/eval_najy_1006_1350_m1_t03_e30"
SETS = [(f"실기 1350 ep{e}", EVAL, e, list(range(0, 21 * 24 + 1, 42))) for e in (0, 1, 2)]
SETS += [(f"학습 task03 ep{e}", "kiroaiseoul/task03_turn_to_face_beaker", e, list(range(0, 61, 10))) for e in (0, 40, 80)]
data = []
for label, repo, e, frames in SETS:
    ds = LeRobotDataset(repo, episodes=[e]); frames = [f for f in frames if f < len(ds)]
    items = [ds[f] for f in frames]
    act = np.stack([it["action"].numpy() for it in items])
    data.append((label, ds.fps, frames, items, act))
res = {}
pol_cache = {}
for name, ck, k, h in CFG:
    if ck not in pol_cache:
        pol_cache.clear(); torch.cuda.empty_cache()
        pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
        pol_cache[ck] = (pol, pre, post)
    pol, pre, post = pol_cache[ck]
    cams = [c for c in pol.config.input_features if c.startswith("observation.images")]
    for label, fps, frames, items, act in data:
        out = []
        for it in items:
            hot = np.zeros(k, np.float32)
            if h is not None: hot[h] = 1
            obs = {"observation.state": np.concatenate([it["observation.state"].numpy()[:14].astype(np.float32), np.zeros(2, np.float32), hot])}
            for c in cams: obs[c] = (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            with torch.inference_mode():
                a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
            out.append((a[:, 15].mean(), np.abs(a[:, 15]).max()))
        res[(name, label)] = np.array(out)
print("청크(30스텝) theta.vel 평균 [rad/s]. 음수 = task03 시연의 회전 방향. 열 = 프레임 시각(s)")
for label, fps, frames, items, act in data:
    print(f"\n== {label} ({fps} fps 표기) | t(s): " + " ".join(f"{f / fps:6.1f}" for f in frames))
    print(f"   {'기록된 action θ':<14}: " + " ".join(f"{v:+6.2f}" for v in act[:, 15]))
    for name, ck, k, h in CFG:
        r = res[(name, label)]
        print(f"   {name:<14}: " + " ".join(f"{v:+6.2f}" for v in r[:, 0]) + f"   | 청크 내 |θ| 최대의 최댓값 {r[:, 1].max():.2f}")
