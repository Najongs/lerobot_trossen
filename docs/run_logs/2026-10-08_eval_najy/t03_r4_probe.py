"""t03: 4라운드·M1 에 실기 1110 프레임과 시연 시작 프레임을 넣어 회전 명령(θ)·p 를 원핫별로 본다. 로봇 무접촉."""
import glob, os, json, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda")
def snap(m): return glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--kiroaiseoul--{m}/snapshots/*/"))[0]
MODELS = [("4라운드 60K", "NAJY_act_all11_r4_27D_60k_s1000", True), ("M1", "NAJY_act_all11_hot_27D_120k_s1000", False)]
HOTS = [("3/11 (맞음)", 2), ("2/11", 1), ("4/11", 3), ("0벡터", None)]
R = "1008_1110_chain_r4_t03~_e30"
SETS = [("실기 1110", f"kiroaiseoul/eval_{R}", 0, [52 + 21 * k for k in (0, 2, 4, 8, 16, 24)]),
        ("시연 t03 ep0", "kiroaiseoul/task03_turn_to_face_beaker", 0, [0, 15, 30, 60]), ("시연 t03 ep40", "kiroaiseoul/task03_turn_to_face_beaker", 40, [0, 15, 30, 60]), ("시연 t03 ep80", "kiroaiseoul/task03_turn_to_face_beaker", 80, [0, 15, 30, 60]),
        ("10/07 1350 ep0 (M1 성공)", "kiroaiseoul/eval_najy_1006_1350_m1_t03_e30", 0, [0, 42, 168, 294, 336])]
data = []
for label, repo, e, frames in SETS:
    ds = LeRobotDataset(repo, episodes=[e]); fr = [f for f in frames if f < len(ds)]; data.append((label, ds.fps, fr, [ds[f] for f in fr]))
print("첫 청크 평균 theta.vel [rad/s] (음수 = 시연 방향) · p. 열 = 프레임 시각(s)")
for name, m, env in MODELS:
    ck = snap(m); pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
    cams = [c for c in pol.config.input_features if c.startswith("observation.images")]
    print(f"\n== {name}")
    for label, fps, frames, items in data:
        off = 52 if "eval_1008" in label or "1110" in label else 0
        print(f"  -- {label} | t: " + " ".join(f"{(f - (52 if '1110' in label else 0)) / fps:5.1f}" for f in frames))
        for hn, h in HOTS:
            out = []
            for it in items:
                hot = np.zeros(11, np.float32)
                if h is not None: hot[h] = 1
                obs = {"observation.state": np.concatenate([it["observation.state"].numpy()[:14].astype(np.float32), np.zeros(2, np.float32), hot])}
                if env: obs["observation.environment_state"] = hot.copy()
                for c in cams: obs[c] = (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                with torch.inference_mode():
                    a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
                out.append((a[:, 15].mean(), a[-1, 16] if a.shape[1] > 16 else float("nan")))
            print(f"     {hn:<11}: " + " ".join(f"{t:+.2f}" + (f" p{p:.2f}" if p == p else "") for t, p in out))
    del pol; torch.cuda.empty_cache()
