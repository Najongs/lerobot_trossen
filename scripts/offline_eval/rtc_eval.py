#!/usr/bin/env python
"""Teacher-forced replay of plan-dependent schemes (RTC guidance, hard replacement).

``simulate.py`` 는 청크가 실행 방식과 무관할 때만 쓸 수 있다. RTC(real-time chunking,
lerobot 0.4.4 ``policies/rtc``)는 새 청크를 "아직 실행 안 한 이전 계획"에 맞춰 디노이징하므로
청크가 실행 이력에 의존한다. 그래서 여기서는 정책을 스텝 순서대로 다시 돌린다.

모형 (record_ensemble 의 비동기 경로와 같은 시간 규칙):
- 스텝 s 에 관측 s 로 요청 -> 스텝 s+L 에 도착 (L = 비행 중 지연). 첫 청크는 로봇이 기다린다.
- 도착한 청크는 앞 L 칸을 잘라내고 그대로 교체한다 (merge=latest). RTC 가 켜져 있으면 그 청크는
  요청 시점의 남은 계획을 prefix 로 받아 디노이징되어, 교체 지점에서 이어지도록 유도된다.
- 요청 간격 R (>= L). R = L 이면 워커가 쉬지 않는 것과 같다.
- 에피소드 여러 개를 배치로 묶어 같은 스텝에 요청한다 (모든 에피소드가 스텝 0 에서 시작).

관측은 시연에서 온다 (teacher-forced). 지표는 simulate.py 와 같다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default="")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--variants", required=True,
                    help="name:rtc(0/1):lag:interval:horizon:guidance:noise;...  예: rtc5:1:5:5:15:10:1.0")
    ap.add_argument("--task", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from predict_chunks import resolve_policy

    dev = torch.device("cuda")
    path = resolve_policy(args.policy)
    root = args.root or None
    meta = LeRobotDatasetMetadata(args.dataset, root=root)
    n = meta.total_episodes
    episodes = sorted({int(round(i * (n - 1) / max(1, args.n_episodes - 1))) for i in range(args.n_episodes)})
    ds = LeRobotDataset(args.dataset, root=root, episodes=episodes)
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    starts = {e: int(np.nonzero(ep_col == e)[0][0]) for e in episodes}
    lens = {e: int((ep_col == e).sum()) for e in episodes}
    gt_all = np.stack([np.asarray(x, dtype=np.float32) for x in ds.hf_dataset["action"]])
    st_all = np.stack([np.asarray(x, dtype=np.float32) for x in ds.hf_dataset["observation.state"]])

    policy = SmolVLAPolicy.from_pretrained(path).to(dev).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path)
    C, D = policy.config.chunk_size, policy.config.max_action_dim
    A = policy.config.action_feature.shape[0]

    rows = []
    for spec in args.variants.split(";"):
        name, rtc, lag, interval, horizon, guidance, ns = spec.split(":")
        rtc, lag, interval, horizon = int(rtc), int(lag), int(interval), int(horizon)
        guidance, ns = float(guidance), float(ns)
        assert interval >= lag >= 0
        if rtc:
            policy.config.rtc_config = RTCConfig(
                enabled=True, execution_horizon=horizon, max_guidance_weight=guidance,
                prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
            )
        else:
            policy.config.rtc_config = None
        policy.init_rtc_processor()
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        E = len(episodes)
        T = max(lens.values())
        plan = [None] * E            # normalised (1, n, A): actions from the current step on
        pending = [None] * E         # (lands_at, request_step, chunk)
        exe = [np.zeros((lens[e], A), np.float32) for e in episodes]
        age = [np.zeros(lens[e], np.float32) for e in episodes]
        plan_obs = [0] * E
        last_req = [-(1 << 30)] * E
        t0 = time.time()
        n_inf = 0

        def infer(idx_list, step):
            nonlocal n_inf
            items = [ds[starts[episodes[i]] + step] for i in idx_list]
            obs = {k: torch.stack([it[k] for it in items]) for k in items[0] if k.startswith("observation.")}
            obs["task"] = [args.task or it["task"] for it in items]
            batch = pre(obs)
            B = len(idx_list)
            noise = torch.randn(B, C, D, device=dev, generator=gen) * ns
            kw = {}
            if rtc:
                left = torch.zeros(B, C, A, device=dev)
                for j, i in enumerate(idx_list):
                    p = plan[i]
                    if p is not None and p.shape[1] > 0:
                        m = min(C, p.shape[1])
                        left[j, :m] = p[0, :m]
                have = any(plan[i] is not None for i in idx_list)
                kw = {"inference_delay": lag, "prev_chunk_left_over": left if have else None,
                      "execution_horizon": horizon}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                a = policy._get_action_chunk(dict(batch), noise, **kw).float()
            n_inf += 1
            return a  # normalised (B, C, A)

        for step in range(T):
            active = [i for i, e in enumerate(episodes) if step < lens[e]]
            # 1. landings
            for i in active:
                if pending[i] is not None and pending[i][0] <= step:
                    _, req, chunk = pending[i]
                    l = step - req
                    plan[i] = chunk[:, l:] if l < chunk.shape[1] else None
                    plan_obs[i] = req
                    pending[i] = None
            # 2. requests (batched across episodes)
            want = [i for i in active if pending[i] is None and step - last_req[i] >= interval]
            if want:
                a = infer(want, step)
                for j, i in enumerate(want):
                    pending[i] = (step + lag, step, a[j : j + 1])
                    last_req[i] = step
            # 3. first step / dry: the robot waits for its in-flight chunk
            for i in active:
                if plan[i] is None or plan[i].shape[1] == 0:
                    if pending[i] is None:
                        a = infer([i], step)
                        pending[i] = (step, step, a)
                        last_req[i] = step
                    _, req, chunk = pending[i]
                    l = step - req
                    plan[i] = chunk[:, l:]
                    plan_obs[i] = req
                    pending[i] = None
                act = plan[i][:, 0]
                plan[i] = plan[i][:, 1:]
                exe[i][step] = post(act.clone()).float().cpu().numpy()[0]
                age[i][step] = step - plan_obs[i]
        per_ep = []
        for i, e in enumerate(episodes):
            s = starts[e]
            per_ep.append(metrics(exe[i], gt_all[s : s + lens[e]], st_all[s : s + lens[e]], age[i]))
        agg = {k: float(np.mean([r[k] for r in per_ep])) for k in per_ep[0]}
        row = {"variant": name, "rtc": rtc, "lag": lag, "interval": interval, "horizon": horizon,
               "guidance": guidance, "noise": ns, "dataset": args.dataset, "task_override": args.task,
               "inferences": n_inf, "seconds": round(time.time() - t0, 1), **agg}
        rows.append(row)
        print(json.dumps(row), flush=True)
        np.savez_compressed(Path(args.out).with_suffix(f".{name}.npz"),
                            **{f"exe_{e}": exe[i] for i, e in enumerate(episodes)})
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
