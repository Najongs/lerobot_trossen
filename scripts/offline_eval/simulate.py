#!/usr/bin/env python
"""Replay ``record_ensemble.py``'s asynchronous scheduling on stored teacher-forced chunks.

``predict_chunks.py`` 가 저장한 청크(프레임마다 하나)를 가지고, 로봇에서 도는 비동기 경로
(``select_action`` 의 async 분기)를 스텝 단위로 그대로 따라 한다. 병합 함수(``_merge``,
``_pop``)와 가중치(``ACTTemporalEnsembler``)는 ``record_ensemble.py`` 와 lerobot 에서 직접
가져다 쓴다. 제어 흐름만 여기서 다시 적었다 -- 원본의 줄 순서를 주석으로 대응시켜 두었다.

워커 모형: 스텝 s 에 제출한 요청은 스텝 s+L 의 poll 에서 도착한다 (L = 비행 중 지연, 로봇 로그의
"비행 중 지연" 값). 워커는 쉬지 않고 다음 요청을 받는다. 에피소드 첫 스텝은 로봇이 기다리므로
지연 0 으로 도착한다.

한계: teacher-forced 다. 관측은 항상 시연에서 오므로, 정책이 틀려서 로봇이 시연 궤적을
벗어나는 복합 오차(closed-loop drift)는 재현하지 않는다. 여기서 재는 것은 "같은 예측들을
실행 방식이 얼마나 뭉개거나 늦추는가"다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from record_ensemble import _crossfade, _merge, _pop  # noqa: E402  (same merge/fade the robot runs)
from lerobot.policies.act.modeling_act import ACTTemporalEnsembler  # noqa: E402

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]
BASE = [14, 15]


def simulate_episode(chunks: np.ndarray, *, lag: int, commit: int, merge: str, coeff: float,
                     every: int = 1, chunk_size: int = 50, fade: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """chunks: (T, 50, D) prediction from the observation at each step. Returns executed (T, D)
    and, per step, the age (steps) of the newest observation that contributed to that action."""
    T = chunks.shape[0]
    ch = torch.from_numpy(chunks).float()

    def new_ens():
        e = ACTTemporalEnsembler(coeff, chunk_size)
        e.reset()
        return e

    ens, pending = new_ens(), new_ens()
    step, last_submit, since_switch = 0, -(1 << 30), 0
    inflight: tuple[int, int] | None = None  # (request_step, lands_at_step)
    out = np.zeros((T, chunks.shape[2]), dtype=np.float32)
    age = np.zeros(T, dtype=np.float32)
    newest_obs = 0  # newest observation step folded into what is being executed

    def has(e):
        return e.ensembled_actions is not None and e.ensembled_actions.shape[1] > 0

    def dry():
        return ens.ensembled_actions is None or ens.ensembled_actions.shape[1] == 0

    def set_(e, a):
        e.ensembled_actions = a.clone()
        e.ensembled_actions_count = torch.ones((a.shape[1], 1), dtype=torch.long)

    def trim(req):
        l = step - req  # align=True
        a = ch[req : req + 1]
        if l >= a.shape[1]:
            return None
        return a[:, l:] if l > 0 else a

    pending_obs = [0]

    def land(req):  # _land
        a = trim(req)
        if a is None:
            return
        if merge == "latest":
            set_(pending, a)
        else:
            _merge(pending, a)
        pending_obs[0] = max(pending_obs[0], req)

    def fold(req):  # _fold
        nonlocal newest_obs
        a = trim(req)
        if a is None:
            return
        if merge == "latest":
            set_(ens, a)
        else:
            _merge(ens, a)
        newest_obs = max(newest_obs, req)

    def switch():  # _switch
        nonlocal since_switch, newest_obs
        new = _crossfade(ens.ensembled_actions, pending.ensembled_actions, merge, fade)
        set_(ens, new)
        pending.ensembled_actions = None
        pending.ensembled_actions_count = None
        since_switch = 0
        newest_obs = pending_obs[0]

    while step < T:
        # 1. poll
        if inflight is not None and step >= inflight[1]:
            req = inflight[0]
            inflight = None
            land(req) if commit else fold(req)
        if commit and has(pending) and (dry() or since_switch >= commit):
            switch()
        # 2. submit
        if inflight is None and step - last_submit >= every:
            inflight = (step, step + lag)
            last_submit = step
        # 3. block while dry (the robot waits; the in-flight result lands "now")
        while dry():
            if commit and has(pending):
                switch()
                continue
            if inflight is None:
                inflight = (step, step)
                last_submit = step
            req = inflight[0]
            inflight = None
            land(req) if commit else fold(req)
        out[step] = _pop(ens)[0].numpy()
        age[step] = step - newest_obs
        step += 1
        since_switch += 1
        if commit and has(pending):
            _pop(pending)
    return out, age


def motion_gain(exe: np.ndarray, gt: np.ndarray, w: int = 25, thr: float = 0.05) -> float:
    """실행 궤적이 시연만큼 움직이는가: w 스텝 창의 변위를 시연 변위에 사영한 비율의 중앙값."""
    if len(gt) <= w:
        return float("nan")
    g = gt[w:, ARM] - gt[:-w, ARM]
    q = exe[w:, ARM] - exe[:-w, ARM]
    n = np.linalg.norm(g, axis=1)
    m = n > thr
    if not m.any():
        return float("nan")
    return float(np.median((g[m] * q[m]).sum(1) / n[m] ** 2))


def metrics(exe: np.ndarray, gt: np.ndarray, state: np.ndarray, age: np.ndarray) -> dict:
    d = exe - gt
    dexe = np.diff(exe[:, ARM], axis=0)
    dgt = np.diff(gt[:, ARM], axis=0)
    return {
        "gain25": motion_gain(exe, gt, 25),
        "gain50": motion_gain(exe, gt, 50),
        "arm_rmse": float(np.sqrt((d[:, ARM] ** 2).mean())),
        "grip_rmse": float(np.sqrt((d[:, GRIP] ** 2).mean())),
        "base_rmse": float(np.sqrt((d[:, BASE] ** 2).mean())),
        "arm_step_rms": float(np.sqrt((dexe ** 2).mean())),  # 스텝 간 변화 (부드러움)
        "arm_step_max": float(np.abs(dexe).max()),
        "arm_jerk_rms": float(np.sqrt((np.diff(exe[:, ARM], 2, axis=0) ** 2).mean())),
        "demo_step_rms": float(np.sqrt((dgt ** 2).mean())),
        "demo_jerk_rms": float(np.sqrt((np.diff(gt[:, ARM], 2, axis=0) ** 2).mean())),
        "clamp_frac": float((np.abs(exe[:, ARM] - state[:, ARM]) > 0.1).any(1).mean()),
        "age_mean": float(age.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--pred", default="all", help="pred_* 키 (쉼표) 또는 all")
    ap.add_argument("--lags", default="3,5")
    ap.add_argument("--schemes", default="default",
                    help="'default' 또는 'commit:merge:coeff[:fade];...' (commit=0 이면 연속 TA)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    preds = [k for k in d.files if k.startswith("pred_")] if args.pred == "all" else [f"pred_{p}" for p in args.pred.split(",")]
    if args.schemes == "default":
        schemes = []
        for c in (0,):
            for cf in (0.01, 0.0, -0.3, -1.0, -1.5):
                schemes.append((c, "average", cf))
        schemes.append((0, "latest", 0.0))
        for c in (5, 10, 15, 20, 30, 40, 50):
            for mg, cf in (("average", -1.5), ("average", 0.0), ("latest", 0.0)):
                schemes.append((c, mg, cf))
    else:
        schemes = []
        for s in args.schemes.split(";"):
            f = s.split(":")
            schemes.append((int(f[0]), f[1], float(f[2]), int(f[3]) if len(f) > 3 else 0))
    lags = [int(x) for x in args.lags.split(",")]

    ep, gt_all, st_all = d["episode"], d["gt"][:, 0], d["state"]
    # 스케줄은 "한 행 = 한 제어 스텝"을 가정한다. --stride>1 로 만든 파일은 시간축이 틀어지므로 거부한다.
    for e in np.unique(ep):
        fr = d["frame"][ep == e]
        if not (np.diff(fr) == 1).all():
            raise SystemExit(f"{args.npz}: episode {e} 의 프레임이 연속이 아니다 (--stride 1 로 만든 파일만 쓸 수 있다)")
    rows = []
    schemes = [s if len(s) == 4 else (*s, 0) for s in schemes]
    for pk, lag, (commit, merge, coeff, fade) in itertools.product(preds, lags, schemes):
        per_ep = []
        for e in np.unique(ep):
            m = ep == e
            exe, age = simulate_episode(d[pk][m], lag=lag, commit=commit, merge=merge, coeff=coeff, fade=fade)
            per_ep.append(metrics(exe, gt_all[m], st_all[m], age))
        agg = {k: float(np.mean([r[k] for r in per_ep])) for k in per_ep[0]}
        rows.append({"pred": pk[5:], "lag": lag, "commit": commit, "merge": merge, "coeff": coeff, "fade": fade, **agg})
        print(json.dumps(rows[-1]), flush=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
