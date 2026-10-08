"""task03 시연 24개의 첫 프레임(정지 시작)에 모델을 넣어 첫 청크 θ 평균을 정답(그 시연의 처음 30프레임 θ 평균)과 대조 — 「정지 시작에서 회전을 내나」. 로봇 무접촉."""
import glob, os, numpy as np, torch, pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda")
def snap(m): return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--kiroaiseoul--{m}/snapshots/*/"))[0]
MODELS = [("4라운드 60K", "NAJY_act_all11_r4_27D_60k_s1000", True), ("3라운드 60K", "NAJY_act_all11_r3_27D_60k_s1000", True), ("M1", "NAJY_act_all11_hot_27D_120k_s1000", False)]
EPS = list(range(0, 117, 5))
ds = LeRobotDataset("kiroaiseoul/task03_turn_to_face_beaker", episodes=EPS)
starts = {}
i = 0
for e in EPS:
    n = ds.meta.episodes[e]["length"] if hasattr(ds.meta, "episodes") and isinstance(ds.meta.episodes, dict) else None
    starts[e] = i
    i += (n if n else 0)
# robust: derive per-episode start indices from episode_index column
epi = np.array([int(x) for x in ds.hf_dataset["episode_index"]])
first = {e: int(np.where(epi == e)[0][0]) for e in EPS}
gt = {}
for e in EPS:
    idx = np.where(epi == e)[0][:30]
    acts = np.stack([np.array(ds.hf_dataset[int(j)]["action"]) for j in idx])
    gt[e] = acts[:, 15].mean()
print(f"시연 {len(EPS)}개 첫 30프레임(1 s) θ 평균의 정답: 중앙 {np.median(list(gt.values())):+.3f}, |θ|>0.10 인 시연 {sum(abs(v)>0.1 for v in gt.values())}/{len(EPS)}")
for name, m, env in MODELS:
    ck = snap(m); pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
    cams = [c for c in pol.config.input_features if c.startswith("observation.images")]
    preds = {}
    for e in EPS:
        it = ds[first[e]]; hot = np.zeros(11, np.float32); hot[2] = 1
        obs = {"observation.state": np.concatenate([it["observation.state"].numpy()[:14].astype(np.float32), np.zeros(2, np.float32), hot])}
        if env: obs["observation.environment_state"] = hot.copy()
        for c in cams: obs[c] = (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        with torch.inference_mode():
            a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
        preds[e] = a[:, 15].mean()
    pv = np.array([preds[e] for e in EPS]); gv = np.array([gt[e] for e in EPS])
    print(f"== {name}: 예측 첫 청크 θ 중앙 {np.median(pv):+.3f} | |θ|>0.10 예측 {int((abs(pv)>0.1).sum())}/{len(EPS)} | 정답이 |θ|>0.10 인 시연 중 예측도 >0.10 인 것 {int(((abs(gv)>0.1)&(abs(pv)>0.1)).sum())}/{int((abs(gv)>0.1).sum())} | 예측/정답 비 중앙 {np.median(pv[abs(gv)>0.05]/gv[abs(gv)>0.05]):.2f}")
    del pol; torch.cuda.empty_cache()
