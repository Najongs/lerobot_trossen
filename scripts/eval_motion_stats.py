#!/usr/bin/env python
"""실기 eval 에피소드의 팔 움직임을 학습 데이터와 대조한다 (로봇에 연결하지 않는다 -- 로컬 데이터셋만 읽는다).

    uv run --no-sync python scripts/eval_motion_stats.py <학습 repo_id> <eval repo_id> [<eval repo_id> ...] [--exec 30]

예)  uv run --no-sync python scripts/eval_motion_stats.py kiroaiseoul/task05_tube_disposal \
         kiroaiseoul/eval_najy_1002_1422_base_t05_e30 --exec 30

에피소드마다 내는 것 (팔 12관절, 그리퍼 제외, observation.state 앞 14칸 기준):
  프레임당 이동      -- 프레임 간 관절 변화 합의 평균 (rad). 학습 데이터는 같은 값을 학습 fps 로 잰다
  방향 바뀜/s        -- 관절 속도 부호가 바뀐 횟수/초, 관절 평균 (떨림 지표)
  시작점 최근접      -- 첫 프레임과 가장 가까운 학습 에피소드 시작점까지 거리. 학습 시작점끼리의
                        같은 거리(중앙값)를 기준으로 함께 출력한다
  청크 경계 점프     -- action 의 프레임 간 변화를 exec 주기의 경계(k=0)와 나머지로 나눈 평균과 그 비
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / ".cache/huggingface/lerobot"
ARM = [i for i in range(14) if i not in (6, 13)]


def load(repo_id):
    d = ROOT / repo_id
    files = sorted(glob.glob(str(d / "data/**/*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"{d} 에 저장된 에피소드가 없다")
    fps = json.load(open(d / "meta/info.json"))["fps"]
    df = pd.concat([pd.read_parquet(f) for f in files])
    return [(ep, g) for ep, g in df.groupby("episode_index")], fps


def arr(g, col):
    return np.stack(g[col].values)[:, :14]


def motion(s, fps):
    v = np.diff(s[:, ARM], axis=0)
    flips = (np.diff(np.sign(v), axis=0) != 0) & (np.abs(v[1:]) * fps > 0.05)
    return np.abs(v).sum(1).mean(), flips.sum(0).mean() / (len(s) / fps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("train")
    p.add_argument("evals", nargs="+")
    p.add_argument("--exec", type=int, default=30, dest="ex")
    a = p.parse_args()

    eps, fps = load(a.train)
    starts = np.stack([arr(g, "observation.state")[0, ARM] for _, g in eps])
    pair = np.linalg.norm(starts[:, None] - starts[None], axis=2)
    np.fill_diagonal(pair, np.inf)
    m = np.array([motion(arr(g, "observation.state"), fps) for _, g in eps])
    print(f"학습 {a.train}: {len(eps)}ep, {fps} fps | 프레임당 이동 중앙값 {np.median(m[:, 0]):.3f} rad, "
          f"방향 바뀜 {np.median(m[:, 1]):.2f}/s | 시작점끼리 최근접 거리 중앙값 {np.median(pair.min(1)):.2f}")

    for r in a.evals:
        reps, efps = load(r)
        for ep, g in reps:
            s, act = arr(g, "observation.state"), arr(g, "action")
            step, flip = motion(s, efps)
            near = np.linalg.norm(starts - s[0, ARM], axis=1).min()
            da = np.abs(np.diff(act[:, ARM], axis=0)).sum(1)
            k = np.arange(1, len(act)) % a.ex
            seam, rest = da[k == 0].mean(), da[k != 0].mean()
            print(f"{r} ep{ep}: {len(s) / efps:.1f}s, {efps} fps | 프레임당 이동 {step:.3f} rad, 방향 바뀜 {flip:.2f}/s | "
                  f"시작점 최근접 {near:.2f} | 청크 경계 점프 {seam:.3f} / 그 외 {rest:.3f} ({seam / rest:.1f}x, exec {a.ex})")


if __name__ == "__main__":
    main()
