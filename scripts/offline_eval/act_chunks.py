#!/usr/bin/env python
"""predict_chunks.py 와 같은 형식으로 ACT 체크포인트의 청크(앞 50스텝)를 저장한다 -- 비교 기준선용.
ACT 는 추론 때 잠재변수를 0 으로 두므로 결정적이다."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True); ap.add_argument("--dataset", required=True)
ap.add_argument("--n-episodes", type=int, default=12); ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
args = ap.parse_args()
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
# ACT 단계 헤드(J52/J53) load hook -- 3차원 action_head 가 없는 체크포인트에는 무동작. 로봇 SDK 를 끌지 않게 파일 경로로 연다
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("act_stage_heads", Path(__file__).resolve().parents[2] / "packages/lerobot_robot_trossen/src/lerobot_robot_trossen/act_stage_heads.py")
_sh = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_sh); _sh.install_load_hook()
path =snapshot_download(args.policy, repo_type="model")
# 상대 EEF 체크포인트(J57/J58)는 출력이 관절이 아니다 -- 관절 GT 와 비교하면 MAE 가 조용히 틀린다.
for _d in (Path(path), *Path(path).resolve().parents[:4]):
    if (_d / "multi_manifest.json").is_file():
        _m = json.loads((_d / "multi_manifest.json").read_text()); _m = _m if isinstance(_m, list) else _m.get("entries", _m)
        if any(e.get("action_space", "joint") != "joint" for e in _m):
            sys.exit(f"{args.policy}: action_space rel_eef 체크포인트 -- 이 도구는 관절 출력만 채점한다")
        break
meta = LeRobotDatasetMetadata(args.dataset); n = meta.total_episodes
episodes = sorted({int(round(i * (n - 1) / max(1, args.n_episodes - 1))) for i in range(args.n_episodes)})
ds = LeRobotDataset(args.dataset, episodes=episodes, delta_timestamps={"action": [i / meta.fps for i in range(50)]})
ep_col = np.asarray(ds.hf_dataset["episode_index"]); fr_col = np.asarray(ds.hf_dataset["frame_index"])
keep = np.nonzero(fr_col % args.stride == 0)[0]
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, keep.tolist()), batch_size=32, num_workers=8)
dev = torch.device("cuda")
pol = ACTPolicy.from_pretrained(path).to(dev).eval()
pre, post = make_pre_post_processors(pol.config, pretrained_path=path)
P, G, PAD, S, E, F = [], [], [], [], [], []
done = 0; t0 = time.time()
for item in loader:
    B = item["action"].shape[0]; idx = keep[done:done + B]
    obs = {k: v for k, v in item.items() if k.startswith("observation.")}
    obs["task"] = list(item["task"])
    batch = pre(obs)
    with torch.inference_mode():
        a = pol.predict_action_chunk(batch)[:, :50]
    P.append(post(a).float().cpu().numpy()); G.append(item["action"].float().numpy()); PAD.append(item["action_is_pad"].numpy())
    S.append(item["observation.state"].float().numpy()); E.append(ep_col[idx]); F.append(fr_col[idx]); done += B
np.savez_compressed(Path(args.out) / f"{args.tag}.npz", pred_act=np.concatenate(P), gt=np.concatenate(G), gt_pad=np.concatenate(PAD),
                    state=np.concatenate(S), episode=np.concatenate(E), frame=np.concatenate(F))
(Path(args.out) / f"{args.tag}.json").write_text(json.dumps({"argv": sys.argv, "policy_path": path, "episodes": episodes, "seconds": round(time.time() - t0, 1)}, indent=1))
print("DONE", done, f"{time.time() - t0:.0f}s")
