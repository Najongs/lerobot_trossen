"""DGX 요청: T8(M1 레시피 + 기본 영상 증강, 2호기 80K) vs M1 — 실기 t01 시작 프레임(1040·1046·1047)과 시연 시작 프레임에 원핫 1/11 을 넣고
첫 청크의 베이스 명령(x/θ)이 회전인가 직진인가. 16D action(진행도 없음). 로봇 무접촉."""
import glob, os, json, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
from huggingface_hub import snapshot_download
dev = torch.device("cuda")
def snap(m):
    d = snapshot_download(m); return d if os.path.exists(os.path.join(d, "config.json")) else os.path.join(d, "pretrained_model")
MODELS = [("M1 (증강 없음)", "kiroaiseoul/NAJY_act_all11_hot_27D_120k_s1000"), ("T8 (기본 증강 on, 80K)", "kiroaiseoul/NAJY_act_all11_hot_aug_80k_t8"), ("4라운드 60K", "kiroaiseoul/NAJY_act_all11_r4_27D_60k_s1000")]
SETS = [("실기 1040", "kiroaiseoul/eval_1008_1040_chain_r4_t01_e30", 0, [52, 52 + 42, 52 + 84]),
        ("실기 1046", "kiroaiseoul/eval_1008_1046_chain_r4_t01_e30", 0, [52, 52 + 42, 52 + 84]),
        ("실기 1047", "kiroaiseoul/eval_1008_1047_chain_r4_t01_e30", 0, [52, 52 + 42, 52 + 84]),
        ("시연 ep0", "kiroaiseoul/task01_move_to_tube_rack", 0, [0, 15, 30]), ("시연 ep50", "kiroaiseoul/task01_move_to_tube_rack", 50, [0, 15, 30]), ("시연 ep99", "kiroaiseoul/task01_move_to_tube_rack", 99, [0, 15, 30])]
data = []
for label, repo, e, frames in SETS:
    ds = LeRobotDataset(repo, episodes=[e]); data.append((label, ds.fps, frames, [ds[f] for f in frames]))
print("첫 청크(30) 평균 x.vel / theta.vel [m/s, rad/s], 원핫 1/11 · 열 = 시작·+2 s·+4 s (시연은 0·0.5·1 s). 시연 시작의 정답은 「θ +, x ≈0」(제자리 회전)")
for name, repo in MODELS:
    ck = snap(repo); cfg = json.load(open(os.path.join(ck, "config.json")))
    dim = cfg["input_features"]["observation.state"]["shape"][0]; k = dim - 16
    env = "observation.environment_state" in cfg["input_features"]; adim = cfg["output_features"]["action"]["shape"][0]
    pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
    cams = [c for c in pol.config.input_features if c.startswith("observation.images")]
    print(f"\n== {name}: state {dim}D (원핫 K={k}) · action {adim}D · ENV {env}")
    for label, fps, frames, items in data:
        out = []
        for it in items:
            hot = np.zeros(k, np.float32); hot[0] = 1
            obs = {"observation.state": np.concatenate([it["observation.state"].numpy()[:14].astype(np.float32), np.zeros(2, np.float32), hot])}
            if env: obs["observation.environment_state"] = hot.copy()
            for c in cams: obs[c] = (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            with torch.inference_mode():
                a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
            out.append((a[:, 14].mean(), a[:, 15].mean()))
        verdict = "회전" if out[0][1] > 0.12 and abs(out[0][0]) < 0.08 else ("직진" if out[0][0] > 0.08 else "정지/약함")
        print(f"   {label:<10}: " + "  ".join(f"{x:+.2f}/{t:+.2f}" for x, t in out) + f"   → 시작 {verdict}")
    del pol; torch.cuda.empty_cache()
