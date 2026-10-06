"""task03 정지 에피소드(1350 ep1·ep2) 프레임에서 무엇을 바꾸면 M1(원핫 3/11)이 회전 명령을 내는가 — 팔 state / cam_high / 손목 카메라를
ep0 의 프레임(회전 전 2·8초 = 베이스가 아직 안 돈 장면, 회전 중 14·16초)과 학습 ep0 2초의 것으로 바꿔 본다. 로봇 무접촉. 섞은 관측은 물리적으로 없는 조합이라 방향만 본다."""
import glob, os, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import prepare_observation_for_inference
dev = torch.device("cuda")
ck = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--kiroaiseoul--NAJY_act_all11_hot_27D_120k_s1000/snapshots/*/"))[0]
pol = ACTPolicy.from_pretrained(ck).to(dev).eval(); pre, post = make_pre_post_processors(pol.config, pretrained_path=ck)
HI, WR = "observation.images.cam_high", ["observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
EVAL, TRAIN = "kiroaiseoul/eval_najy_1006_1350_m1_t03_e30", "kiroaiseoul/task03_turn_to_face_beaker"
def get(repo, e, f):
    it = LeRobotDataset(repo, episodes=[e])[f]
    return {"s": it["observation.state"].numpy()[:14].astype(np.float32), **{c: (it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for c in [HI] + WR}}
def theta(o):
    hot = np.zeros(11, np.float32); hot[2] = 1
    obs = {"observation.state": np.concatenate([o["s"], np.zeros(2, np.float32), hot]), **{c: o[c] for c in [HI] + WR}}
    with torch.inference_mode():
        a = post(pol.predict_action_chunk(pre(prepare_observation_for_inference(obs, dev, None, None))))[0].float().cpu().numpy()
    return a[:, 15].mean()
targets = {f"ep{e}@{t}s": get(EVAL, e, t * 21) for e in (1, 2) for t in (2, 10, 20)}
donors = {"1350 ep0@2s (회전 전)": get(EVAL, 0, 2 * 21), "1350 ep0@8s (회전 전)": get(EVAL, 0, 8 * 21), "1350 ep0@14s": get(EVAL, 0, 14 * 21), "1350 ep0@16s": get(EVAL, 0, 16 * 21), "학습 ep0@2s": get(TRAIN, 0, 60)}
print("M1 원핫 3/11, 청크 theta.vel 평균 [rad/s] (음수 = 회전). 대상 = 정지 프레임, 바꿔 넣는 쪽 = 회전 직전 프레임")
print("대상 원본: " + " ".join(f"{k} {theta(v):+.2f}" for k, v in targets.items()))
print("제공 원본: " + " ".join(f"{k} {theta(v):+.2f}" for k, v in donors.items()))
SW = {"팔 state 만": ["s"], "cam_high 만": [HI], "손목 2개만": WR, "카메라 3개": [HI] + WR, "state+손목": ["s"] + WR, "state+cam_high": ["s", HI]}
for dn, d in donors.items():
    print(f"-- 제공 {dn}")
    for sn, keys in SW.items():
        r = [theta({**t, **{k: d[k] for k in keys}}) for t in targets.values()]
        print(f"   {sn:<14}: " + " ".join(f"{v:+.2f}" for v in r) + f"   | 평균 {np.mean(r):+.2f}")
print("팔 12관절 차이(도, 대상 − 1350 ep0@14s): ")
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]; NM = ["L0", "L1", "L2", "L3", "L4", "L5", "R0", "R1", "R2", "R3", "R4", "R5"]
for k, t in targets.items():
    df = np.degrees(t["s"][ARM] - donors["1350 ep0@14s"]["s"][ARM]); print(f"   {k}: " + " ".join(f"{n}{v:+.0f}" for n, v in zip(NM, df)))
