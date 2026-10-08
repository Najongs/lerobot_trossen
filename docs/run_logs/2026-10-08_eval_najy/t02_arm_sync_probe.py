"""task02 시작 프레임에서 모델별(M1·3라운드·4라운드) 첫 청크의 왼팔/오른팔 움직임과 그리퍼 열림 — 「양팔 동시」 가 어느 레시피에서 생기나. 로봇 무접촉."""
import glob, os, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda")
def snap(m): return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--kiroaiseoul--{m}/snapshots/*/"))[0]
MODELS = [("M1 (1라운드)", "NAJY_act_all11_hot_27D_120k_s1000", False), ("3라운드 60K", "NAJY_act_all11_r3_27D_60k_s1000", True), ("4라운드 60K", "NAJY_act_all11_r4_27D_60k_s1000", True)]
L = [0, 1, 2, 3, 4, 5]; R = [7, 8, 9, 10, 11, 12]
SETS = [("시연 ep0", "kiroaiseoul/task02_pickup_tubes", 0, 0), ("시연 ep44", "kiroaiseoul/task02_pickup_tubes", 44, 0), ("시연 ep88", "kiroaiseoul/task02_pickup_tubes", 88, 0), ("시연 ep132", "kiroaiseoul/task02_pickup_tubes", 132, 0),
        ("실기 1101 시작", "kiroaiseoul/eval_1008_1101_chain_r4_t02_e30", 0, 52), ("실기 1102 시작", "kiroaiseoul/eval_1008_1102_chain_r4_t02_e30", 0, 52)]
items = []
for label, repo, e, f in SETS:
    ds = LeRobotDataset(repo, episodes=[e]); items.append((label, ds[f]))
print("첫 청크(30스텝) 안에서 팔이 시작 action 에서 얼마나 움직이나 (L2, rad): 왼팔 / 오른팔 · 그리퍼 명령 최대 L/R · (p)")
print("시연 기준: 왼팔이 먼저(1.3 s), 오른팔은 7 s 뒤 — 첫 청크(1.4 s)에서 오른팔은 거의 0 이어야 한다")
for name, m, env in MODELS:
    ck = snap(m); pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
    cams = [c for c in pol.config.input_features if c.startswith("observation.images")]
    row = []
    for label, it in items:
        hot = np.zeros(11, np.float32); hot[1] = 1
        obs = {"observation.state": np.concatenate([it["observation.state"].numpy()[:14].astype(np.float32), np.zeros(2, np.float32), hot])}
        if env: obs["observation.environment_state"] = hot.copy()
        for c in cams: obs[c] = (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        with torch.inference_mode():
            a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
        ml = np.linalg.norm(a[-1, L] - a[0, L]); mr = np.linalg.norm(a[-1, R] - a[0, R])
        p = a[-1, 16] if a.shape[1] > 16 else float("nan")
        row.append(f"{label}: {ml:.2f}/{mr:.2f} g{a[:,6].max():.3f}/{a[:,13].max():.3f}" + (f" p{p:.2f}" if a.shape[1] > 16 else ""))
    print(f"\n== {name}\n   " + "\n   ".join(row))
    del pol; torch.cuda.empty_cache()
