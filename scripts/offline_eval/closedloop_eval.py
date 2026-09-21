#!/usr/bin/env python
"""Closed-loop proxy on the demonstration manifold ("재생 폐루프").

teacher-forced 재생(simulate.py)은 state 를 시연에서 가져오므로, 정책이 느리거나 멈춰도 로봇은
시연대로 나아간 것으로 친다. 여기서는 로봇 state 가 **실행한 액션을 따라가게** 한다.
이미지는 렌더링할 수 없으니, 시연 에피소드에서 **현재 로봇 state 와 가장 가까운 프레임**을 찾아
그 프레임의 카메라 3장을 넣는다. 로봇이 시연 경로 근처에 있는 한 관측이 일관되고,
매칭된 프레임 번호 k 가 곧 과제 진행도다 (k 가 끝에 닿으면 "완주").

모형:
- 팔·그리퍼 state(0..13) = 직전 스텝에 실행한 액션 (위치 제어가 한 스텝 안에 따라간다고 봄).
  ``--clamp`` (기본 0.1 rad) 는 로봇의 max_relative_target 과 같은 클램프를 팔 관절에 건다.
- 베이스 state(14,15) = 직전 베이스 명령. 베이스 위치는 이미지에 반영할 수 없으므로 베이스가
  주로 움직이는 과제(task06 의 이동 구간)에는 이 근사가 맞지 않는다. task02 용이다.
- 프레임 매칭: 창 [k-3, k+15] 안에서 정규화된 팔+그리퍼 state 거리가 최소인 프레임. 시연이 멈춘
  구간(state 가 같은 프레임들)에서는 최솟값에서 ``--match-eps`` 안의 후보 중 가장 뒤 프레임을 고른다.
- 실행 스케줄: record_ensemble 의 비동기 경로 (simulate.py 와 같은 규칙, 같은 _merge/_crossfade).
- 샘플링: record_ensemble._predict_chunk 와 같은 방식 (K, noise_scale, reduce).

지표 (에피소드 평균):
- progress: 시간 예산(시연 길이 x ``--budget``) 끝의 k / (길이-1)
- done: k 가 길이의 95% 에 닿은 에피소드 비율, t_done: 닿은 스텝 / 시연 길이
- stall: 30스텝 창에서 k 가 2 미만 전진한 스텝 비율
- jerk, step_max: 실행 궤적의 부드러움
- off: 로봇 state 와 매칭 프레임 state 사이의 평균 거리 (rad, 팔) -- 클수록 시연 경로를 벗어남
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from record_ensemble import _crossfade, _merge, _pop, _reduce_samples  # noqa: E402
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler  # noqa: E402

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
MATCH = ARM + [6, 13]


class Sched:
    """One episode's copy of record_ensemble's async select_action, fed chunks from outside."""

    def __init__(self, lag, commit, merge, coeff, fade, chunk_size=50):
        self.lag, self.commit, self.merge, self.fade, self.coeff, self.C = lag, commit, merge, fade, coeff, chunk_size
        self.ens, self.pending = self._new(), self._new()
        self.step, self.since_switch = 0, 0
        self.inflight = None  # (request_step, lands_at, chunk)

    def _new(self):
        e = ACTTemporalEnsembler(self.coeff, self.C)
        e.reset()
        return e

    @staticmethod
    def _has(e):
        return e.ensembled_actions is not None and e.ensembled_actions.shape[1] > 0

    def dry(self):
        return not self._has(self.ens)

    def _set(self, e, a):
        e.ensembled_actions = a.clone()
        e.ensembled_actions_count = torch.ones((a.shape[1], 1), dtype=torch.long)

    def _trim(self, req, chunk):
        l = self.step - req
        if l >= chunk.shape[1]:
            return None
        return chunk[:, l:] if l > 0 else chunk

    def _accept(self, req, chunk):
        a = self._trim(req, chunk)
        if a is None:
            return
        target = self.pending if self.commit else self.ens
        if self.merge == "latest":
            self._set(target, a)
        else:
            _merge(target, a)

    def _switch(self):
        self._set(self.ens, _crossfade(self.ens.ensembled_actions, self.pending.ensembled_actions, self.merge, self.fade))
        self.pending.ensembled_actions = None
        self.pending.ensembled_actions_count = None
        self.since_switch = 0

    def poll(self):
        """Phase 1 of select_action. Returns True if a new request should be submitted now."""
        if self.inflight is not None and self.inflight[1] <= self.step and self.inflight[2] is not None:
            req, _, chunk = self.inflight
            self.inflight = None
            self._accept(req, chunk)
        if self.commit and self._has(self.pending) and (self.dry() or self.since_switch >= self.commit):
            self._switch()
        return self.inflight is None

    def submit(self, chunk):
        self.inflight = (self.step, self.step + self.lag, chunk)

    def needs_block(self):
        return self.dry() and not (self.commit and self._has(self.pending))

    def resolve_block(self, chunk_now=None):
        """Phase 3: robot waits. The in-flight chunk lands now (or a fresh one if none in flight)."""
        while self.dry():
            if self.commit and self._has(self.pending):
                self._switch()
                continue
            if self.inflight is None:
                self.inflight = (self.step, self.step, chunk_now)
            req, _, chunk = self.inflight
            self.inflight = None
            self._accept(req, chunk)

    def pop(self):
        a = _pop(self.ens)[0]
        self.step += 1
        self.since_switch += 1
        if self.commit and self._has(self.pending):
            _pop(self.pending)
        return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--variants", required=True,
                    help="name:lag:commit:merge:coeff:fade:K:noise:reduce;...")
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--clamp", type=float, default=0.1)
    ap.add_argument("--match-eps", type=float, default=0.05, help="정규화 거리 허용폭 (멈춤 구간 통과용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", default="",
                    help="청크에 더할 관절별 상수(실제 단위), 'dim:val,dim:val'. 예: 1:0.014,8:0.009 (어깨 편향 보정)")
    ap.add_argument("--perfect", action="store_true", help="정책 대신 매칭 프레임의 시연 청크를 쓴다 (검증용)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from predict_chunks import resolve_policy

    dev = torch.device("cuda")
    path = resolve_policy(args.policy)
    meta = LeRobotDatasetMetadata(args.dataset)
    n = meta.total_episodes
    episodes = sorted({int(round(i * (n - 1) / max(1, args.n_episodes - 1))) for i in range(args.n_episodes)})
    ds = LeRobotDataset(args.dataset, episodes=episodes)
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    starts = {e: int(np.nonzero(ep_col == e)[0][0]) for e in episodes}
    lens = {e: int((ep_col == e).sum()) for e in episodes}
    st_all = np.stack([np.asarray(x, dtype=np.float32) for x in ds.hf_dataset["observation.state"]])
    act_all = np.stack([np.asarray(x, dtype=np.float32) for x in ds.hf_dataset["action"]])

    policy = SmolVLAPolicy.from_pretrained(path).to(dev).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path)
    C, D = policy.config.chunk_size, policy.config.max_action_dim
    std = st_all.std(0) + 1e-6
    img_keys = [k for k in policy.config.input_features if "images" in k]
    cache: dict[int, dict] = {}

    def frame_images(i):
        if i not in cache:
            it = ds[i]
            cache[i] = {k: it[k] for k in img_keys}
            cache[i]["task"] = it["task"]
            if len(cache) > 4000:
                cache.pop(next(iter(cache)))
        return cache[i]

    offset = {int(x.split(":")[0]): float(x.split(":")[1]) for x in args.offset.split(",") if x}
    rows = []
    for spec in args.variants.split(";"):
        name, lag, commit, merge, coeff, fade, K, ns, reduce = spec.split(":")
        lag, commit, fade, K = int(lag), int(commit), int(fade), int(K)
        coeff, ns = float(coeff), float(ns)
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        E = len(episodes)
        T = {e: int(lens[e] * args.budget) for e in episodes}
        sch = [Sched(lag, commit, merge, coeff, fade, C) for _ in episodes]
        sim_state = [st_all[starts[e]].copy() for e in episodes]
        k = [0] * E
        k_hist = [[] for _ in episodes]
        exe = [[] for _ in episodes]
        off = [[] for _ in episodes]
        drift = [[] for _ in episodes]  # 부호 있는 관절별 이탈 (로봇 - 매칭 프레임), 0..13
        n_inf = 0
        t0 = time.time()

        def infer(ids):
            nonlocal n_inf
            if args.perfect:
                out = []
                for i in ids:
                    e = episodes[i]
                    s = starts[e] + k[i]
                    idx = np.clip(np.arange(s, s + C), starts[e], starts[e] + lens[e] - 1)
                    out.append(torch.from_numpy(act_all[idx])[None])
                return out
            imgs = [frame_images(starts[episodes[i]] + k[i]) for i in ids]
            obs = {kk: torch.stack([im[kk] for im in imgs]) for kk in img_keys}
            obs["observation.state"] = torch.from_numpy(np.stack([sim_state[i] for i in ids]))
            obs["task"] = [im["task"] for im in imgs]
            batch = pre(obs)
            B = len(ids)
            b = {kk: (v.repeat_interleave(K, 0) if torch.is_tensor(v) and v.shape[0] == B else v) for kk, v in batch.items()}
            noise = torch.randn(B * K, C, D, device=dev, generator=gen) * ns
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                a = policy._get_action_chunk(dict(b), noise).float()
            a = a.view(B, K, *a.shape[1:])
            a = torch.cat([_reduce_samples(a[j], reduce) for j in range(B)])
            a = post(a).float().cpu()  # real units; the ensembler is linear so this commutes
            for dim, val in offset.items():
                a[..., dim] += val
            n_inf += 1
            return [a[j : j + 1] for j in range(B)]

        for t in range(max(T.values())):
            active = [i for i, e in enumerate(episodes) if t < T[e]]
            if not active:
                break
            want = [i for i in active if sch[i].poll()]
            if want:
                for i, c in zip(want, infer(want)):
                    sch[i].submit(c)
            blocked = [i for i in active if sch[i].needs_block()]
            fresh = {}
            if blocked:
                need = [i for i in blocked if sch[i].inflight is None]
                if need:
                    fresh = dict(zip(need, infer(need)))
            for i in active:
                if sch[i].needs_block():
                    sch[i].resolve_block(fresh.get(i))
                a = sch[i].pop().numpy()
                s = sim_state[i]
                if args.clamp > 0:
                    a[ARM] = np.clip(a[ARM], s[ARM] - args.clamp, s[ARM] + args.clamp)
                exe[i].append(a.copy())
                s = a.copy()  # position control follows within a step; base state = base command
                sim_state[i] = s
                e = episodes[i]
                lo, hi = max(0, k[i] - 3), min(lens[e] - 1, k[i] + 15)
                seg = st_all[starts[e] + lo : starts[e] + hi + 1]
                dist = np.linalg.norm((seg[:, MATCH] - s[MATCH]) / std[MATCH], axis=1)
                # 시연이 멈춘 구간은 state 가 같아 argmin 이 늘 멈춤 첫 프레임을 고른다 -> 같은 멈춤을
                # 끝없이 되풀이한다. 최솟값에서 eps 안의 후보 중 가장 뒤 프레임을 고른다.
                k[i] = lo + int(np.nonzero(dist <= dist.min() + args.match_eps)[0].max())
                k_hist[i].append(k[i])
                off[i].append(float(np.linalg.norm(st_all[starts[e] + k[i]][ARM] - s[ARM])))
                drift[i].append(s[:14] - st_all[starts[e] + k[i]][:14])
        per = []
        for i, e in enumerate(episodes):
            kh = np.array(k_hist[i])
            L = lens[e]
            done_at = np.nonzero(kh >= 0.95 * (L - 1))[0]
            x = np.array(exe[i])[:, ARM]
            adv = kh[30:] - kh[:-30] if len(kh) > 30 else np.array([0])
            per.append({
                "episode": int(e), "progress": float(kh[-1] / (L - 1)), "max_progress": float(kh.max() / (L - 1)),
                "done": bool(len(done_at)), "t_done": float(done_at[0] / L) if len(done_at) else float("nan"),
                "stall": float((adv < 2).mean()), "jerk": float(np.sqrt((np.diff(x, 2, axis=0) ** 2).mean())),
                "step_max": float(np.abs(np.diff(x, axis=0)).max()), "off": float(np.mean(off[i])),
                "drift": [float(v) for v in np.mean(drift[i], axis=0)],
            })
        agg = {kk: float(np.nanmean([p[kk] for p in per])) for kk in per[0] if kk not in ("episode", "drift")}
        agg["drift"] = [float(v) for v in np.mean([p["drift"] for p in per], axis=0)]
        row = {"variant": name, "lag": lag, "commit": commit, "merge": merge, "coeff": coeff, "fade": fade,
               "K": K, "noise": ns, "reduce": reduce, "perfect": args.perfect, "dataset": args.dataset,
               "offset": args.offset,
               "inferences": n_inf, "seconds": round(time.time() - t0, 1), **agg, "per_episode": per}
        rows.append(row)
        print(json.dumps({kk: v for kk, v in row.items() if kk != "per_episode"}), flush=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
