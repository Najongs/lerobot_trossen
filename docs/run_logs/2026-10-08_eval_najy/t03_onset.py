"""task03 시연: 회전이 언제 시작되나(정답), 그 전에 팔은 움직이나 — 그리고 정지 시작 프레임에서 모델의 첫 청크가 팔을 움직이나. 로봇 무접촉."""
import glob, os, numpy as np, torch, pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda")
def snap(m): return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--kiroaiseoul--{m}/snapshots/*/"))[0]
ARM = [0,1,2,3,4,5,7,8,9,10,11,12]
C = "/home/trossen-ai/.cache/huggingface/lerobot/kiroaiseoul/"
d = pd.concat([pd.read_parquet(p, columns=['episode_index','frame_index','action','observation.state']) for p in sorted(glob.glob(C+'task03_turn_to_face_beaker/data/*/*.parquet'))]).sort_values(['episode_index','frame_index'])
onset=[]; arm1=[]; arm_before=[]
for e,g in d.groupby('episode_index'):
    a=np.stack(g['action'].to_numpy()); s=np.stack(g['observation.state'].to_numpy())
    big=np.where(np.abs(a[:,15])>0.10)[0]; on=int(big[0]) if len(big) else None
    onset.append(on/30 if on is not None else np.nan)
    arm1.append(np.linalg.norm(np.diff(s[:30,ARM],axis=0),axis=1).sum())
    if on: arm_before.append(np.linalg.norm(s[on,ARM]-s[0,ARM]))
onset=np.array(onset)
print(f"시연 117ep: 회전(|θ|>0.1) 시작 시각(표기 30fps 기준) 중앙 {np.nanmedian(onset):.1f}s · p10 {np.nanpercentile(onset,10):.1f} · p90 {np.nanpercentile(onset,90):.1f} · 1 s 안에 시작 {int((onset<1).sum())}/117 · 2 s 안 {int((onset<2).sum())}/117")
print(f"  첫 1 s(30프레임) 팔 이동 합 중앙 {np.median(arm1):.2f} rad | 회전 시작 전까지 팔이 시작 자세에서 벗어난 거리 중앙 {np.median(arm_before):.2f} rad")
EPS = list(range(0, 117, 5))
ds = LeRobotDataset("kiroaiseoul/task03_turn_to_face_beaker", episodes=EPS)
epi = np.array([int(x) for x in ds.hf_dataset["episode_index"]]); first = {e: int(np.where(epi == e)[0][0]) for e in EPS}
for name, m, env in [("4라운드 60K", "NAJY_act_all11_r4_27D_60k_s1000", True), ("M1", "NAJY_act_all11_hot_27D_120k_s1000", False)]:
    ck = snap(m); pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
    cams = [c for c in pol.config.input_features if c.startswith("observation.images")]
    arms=[]; ths=[]
    for e in EPS:
        it = ds[first[e]]; hot = np.zeros(11, np.float32); hot[2] = 1
        obs = {"observation.state": np.concatenate([it["observation.state"].numpy()[:14].astype(np.float32), np.zeros(2, np.float32), hot])}
        if env: obs["observation.environment_state"] = hot.copy()
        for c in cams: obs[c] = (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        with torch.inference_mode():
            a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
        arms.append(np.linalg.norm(a[-1,ARM]-a[0,ARM])); ths.append(np.abs(a[:,15]).max())
    print(f"== {name}: 시연 정지 시작 프레임 24개 → 첫 청크 팔 이동 중앙 {np.median(arms):.2f} rad (시연 첫 1 s 팔 이동 중앙 {np.median(arm1):.2f}) | 청크 안 |θ| 최대 중앙 {np.median(ths):.2f}")
    del pol; torch.cuda.empty_cache()
