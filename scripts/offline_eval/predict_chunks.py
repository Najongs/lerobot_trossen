#!/usr/bin/env python
"""Teacher-forced chunk predictions for every frame of selected demonstration episodes.

로봇 없이 "이 체크포인트가 시연과 얼마나 다르게 움직이려 하는가"를 재는 첫 단계다.
시연 에피소드의 매 프레임 관측(카메라 3대 + state + task 문장)을 정책에 넣고, 예측 청크
(50 x 16, 실제 단위)를 시연자의 실제 다음 50 액션과 함께 저장한다.

teacher-forced 재생에서는 관측이 시연에서 오므로, 청크 예측은 실행 방식(commit, coeff,
merge, lag)과 무관하다. 그래서 GPU 는 여기서 한 번만 쓰고, 실행 방식 비교는
``simulate.py`` 가 저장된 청크로 CPU 에서 한다.

추론 경로는 로봇과 같다: 체크포인트의 preprocessor -> ``_get_action_chunk`` (bf16 autocast)
-> postprocessor. 샘플링은 ``record_ensemble._predict_chunk`` 와 같은 방식(노이즈 K 개를
배치로, ``noise_scale`` 배, 평균)이다.

한 번의 디코딩 패스에서 여러 샘플링 설정을 같이 돌린다 (``--configs``).

출력: ``<out>/<tag>.npz``
    pred_<cfg>  (N, 50, 16) float32  설정별 예측 청크 (실제 단위)
    gt          (N, 50, 16) float32  시연 액션 a[t..t+49]
    gt_pad      (N, 50)     bool     에피소드 끝을 넘어간 칸
    state       (N, 16)     float32
    episode     (N,)        int      에피소드 번호
    frame       (N,)        int      에피소드 안 프레임 번호
그리고 ``<out>/<tag>.json`` 에 실행 명령·설정·git SHA·체크포인트 경로.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from record_ensemble import _reduce_samples  # noqa: E402  (the robot's own reduction)


def parse_configs(spec: str) -> list[dict]:
    """``name:steps:noise:K[:reduce],name2:...`` -> list of dicts. reduce = mean(기본)|medoid."""
    out = []
    for item in spec.split(","):
        f = item.split(":")
        name, steps, ns, k = f[:4]
        reduce = f[4] if len(f) > 4 else "mean"
        out.append({"name": name, "num_steps": int(steps), "noise_scale": float(ns), "samples": int(k),
                    "reduce": reduce})
    return out


def resolve_policy(path: str) -> str:
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    root = snapshot_download(path, repo_type="model", allow_patterns=["pretrained_model/*"])
    return str(Path(root) / "pretrained_model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="HF repo id 또는 pretrained_model 경로")
    ap.add_argument("--dataset", required=True, help="예: kiroaiseoul/task02_pickup_tubes")
    ap.add_argument("--episodes", default="", help="쉼표 목록. 비우면 --n-episodes 개를 고르게 뽑는다")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--stride", type=int, default=1, help="프레임 간격 (1 = 모든 프레임)")
    ap.add_argument("--configs", default="stock:10:1.0:1")
    ap.add_argument("--task", default="", help="task 문장을 이것으로 덮어쓴다 (프롬프트 민감도 실험)")
    ap.add_argument("--state-shift", type=int, default=0,
                    help="state 를 k 프레임 과거 것으로 바꿔 넣는다 (state 의존도 실험)")
    ap.add_argument("--zero-state-dims", default="",
                    help="state 의 이 차원들(쉼표)을 0 으로 가려 넣는다. 예: 14,15 = 베이스 속도 (따라 하기 실험)")
    ap.add_argument("--batch", type=int, default=16, help="한 번에 넣는 프레임 수 (x K 가 실제 배치)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    ap.add_argument("--root", default="", help="로컬 데이터셋 경로 (허브 버전 태그가 없는 데이터셋용)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = parse_configs(args.configs)
    policy_path = resolve_policy(args.policy)
    dev = torch.device("cuda")

    root = args.root or None
    meta = LeRobotDatasetMetadata(args.dataset, root=root)
    fps = meta.fps
    if args.episodes:
        episodes = [int(e) for e in args.episodes.split(",")]
    else:
        n = meta.total_episodes
        episodes = sorted({int(round(i * (n - 1) / max(1, args.n_episodes - 1))) for i in range(args.n_episodes)})
    ds = LeRobotDataset(
        args.dataset,
        root=root,
        episodes=episodes,
        delta_timestamps={"action": [i / fps for i in range(50)]},
    )

    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    fr_col = np.asarray(ds.hf_dataset["frame_index"])
    keep = np.nonzero(fr_col % args.stride == 0)[0]
    subset = torch.utils.data.Subset(ds, keep.tolist())
    loader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch, num_workers=args.workers, shuffle=False,
        persistent_workers=False, prefetch_factor=4 if args.workers else None,
    )

    # state 의존도 실험용: 같은 에피소드의 k 프레임 전 state (에피소드 시작에서는 첫 프레임으로 고정)
    state_all = None
    if args.state_shift:
        state_all = torch.stack([torch.as_tensor(x) for x in ds.hf_dataset["observation.state"]]).float()

    policy = SmolVLAPolicy.from_pretrained(policy_path).to(dev).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=policy_path)
    C, D = policy.config.chunk_size, policy.config.max_action_dim
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    preds = {c["name"]: [] for c in configs}
    gts, pads, states, eps, frames = [], [], [], [], []
    t0 = time.time()
    done = 0
    for bi, item in enumerate(loader):
        B = item["action"].shape[0]
        idx = keep[done : done + B]
        obs = {k: v for k, v in item.items() if k.startswith("observation.")}
        if state_all is not None:
            src = []
            for i in idx:
                j = max(int(i) - args.state_shift, int(i) - int(fr_col[i]))
                src.append(j)
            obs["observation.state"] = state_all[src]
        if args.zero_state_dims:
            obs["observation.state"] = obs["observation.state"].clone()
            obs["observation.state"][:, [int(x) for x in args.zero_state_dims.split(",")]] = 0.0
        obs["task"] = [args.task] * B if args.task else list(item["task"])
        batch = pre(obs)
        for c in configs:
            policy.config.num_steps = c["num_steps"]
            K = c["samples"]
            b = {
                k: (v.repeat_interleave(K, dim=0) if torch.is_tensor(v) and v.shape[0] == B else v)
                for k, v in batch.items()
            }
            noise = torch.randn(B * K, C, D, device=dev, generator=gen) * c["noise_scale"]
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp == "bf16" else torch.autocast("cuda", enabled=False)
            with torch.inference_mode(), ctx:
                a = policy._get_action_chunk(dict(b), noise).float()
            a = a.view(B, K, *a.shape[1:])
            # 로봇과 같은 함수로 K 개를 하나로 줄인다 (record_ensemble._reduce_samples)
            a = torch.cat([_reduce_samples(a[i], c["reduce"]) for i in range(B)])
            a = post(a)
            preds[c["name"]].append(a.float().cpu().numpy())
        gts.append(item["action"].float().numpy())
        pads.append(item["action_is_pad"].numpy())
        states.append(item["observation.state"].float().numpy())
        eps.append(ep_col[idx])
        frames.append(fr_col[idx])
        done += B
        if bi % 20 == 0:
            el = time.time() - t0
            print(f"[{args.tag}] {done}/{len(keep)} frames  {el:.0f}s  eta {el / done * (len(keep) - done):.0f}s", flush=True)

    np.savez_compressed(
        out_dir / f"{args.tag}.npz",
        gt=np.concatenate(gts), gt_pad=np.concatenate(pads), state=np.concatenate(states),
        episode=np.concatenate(eps), frame=np.concatenate(frames),
        **{f"pred_{k}": np.concatenate(v) for k, v in preds.items()},
    )
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        sha = "unknown"
    (out_dir / f"{args.tag}.json").write_text(json.dumps({
        "argv": sys.argv, "git_sha": sha, "policy_path": policy_path, "dataset": args.dataset,
        "episodes": episodes, "fps": fps, "configs": configs, "frames": int(len(keep)),
        "seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
        "task_override": args.task, "state_shift": args.state_shift, "zero_state_dims": args.zero_state_dims,
        "tasks_in_dataset": sorted(set(meta.tasks.index.tolist())) if hasattr(meta.tasks, "index") else None,
    }, ensure_ascii=False, indent=1))
    print(f"[{args.tag}] DONE {len(keep)} frames in {time.time() - t0:.0f}s -> {out_dir / (args.tag + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
