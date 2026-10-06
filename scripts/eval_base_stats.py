"""베이스 명령·실측 적분 — 회차의 basevel.csv 에서 정책 구간(에피소드)별 회전·전진량을 재고 학습 시연과 비교한다.

    uv run --no-sync python scripts/eval_base_stats.py <회차...> [--train kiroaiseoul/<학습 데이터셋>]

로봇 무접촉 — ~/eval_logs/<회차>.basevel.csv 와 로컬 HF 캐시의 parquet 만 읽는다 (내려받지 않음).

구간 = phase=policy 행을 1초 넘는 틈으로 자른 것. 에피소드 리셋은 teleop 행이라 자동으로 빠지지만,
시작 직후 ← 로 버린 시도도 구간 하나로 잡힌다 — 길이로 가려 읽는다.
적분 = Σ 속도 × 틱 간격(실측). 회전은 theta.vel 을 rad/s 로 보고 도(°)로, 전진은 x.vel 을 m/s 로 보고 m 로.

학습 쪽은 action 의 x.vel·theta.vel 을 **데이터에 적힌 fps** 로 나눠 적분한다. 실제 녹화 주기가 21 Hz 였다면
그 값은 fps/21 배 커야 한다 — 두 값을 같이 찍는다 (어느 쪽이 맞는지는 녹화 로그로만 안다).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

LOGDIR = os.path.expanduser("~/eval_logs")
CACHE = os.path.expanduser("~/.cache/huggingface/lerobot")
GAP_S = 1.0
X, TH = 14, 15  # 16D action 의 x.vel, theta.vel


def run_segments(run: str) -> None:
    path = f"{LOGDIR}/{run}.basevel.csv"
    if not os.path.exists(path):
        print(f"== {run}: basevel.csv 없음 ({path})")
        return
    b = pd.read_csv(path)
    p = b[b["phase"] == "policy"]
    if p.empty:
        print(f"== {run}: policy 구간 없음 (행 {len(b)})")
        return
    t = p["t_mono"].to_numpy()
    seg = np.concatenate([[0], np.cumsum(np.diff(t) > GAP_S)])
    print(f"== {run}: policy 행 {len(p)}, 구간 {seg.max() + 1}개")
    for s in np.unique(seg):
        q = p[seg == s]
        tt = q["t_mono"].to_numpy()
        dt = np.diff(tt)
        dt = np.append(dt, np.median(dt) if len(dt) else 0.0)
        dur = tt[-1] - tt[0] + (dt[-1] if len(dt) else 0.0)
        cmd_th = np.degrees((q["cmd_theta_vel"].to_numpy() * dt).sum())
        meas_th = np.degrees((q["meas_theta_vel"].to_numpy() * dt).sum())
        cmd_x = (q["cmd_x_vel"].to_numpy() * dt).sum()
        meas_x = (q["meas_x_vel"].to_numpy() * dt).sum()
        print(
            f"seg{s}: {dur:5.1f}s {len(q)}행 | 회전 명령 {cmd_th:+6.1f}° 실측 {meas_th:+6.1f}° | "
            f"전진 명령 {cmd_x:+.3f} m 실측 {meas_x:+.3f} m | "
            f"|cmd x| 최대 {q['cmd_x_vel'].abs().max():.3f} |cmd θ| 최대 {q['cmd_theta_vel'].abs().max():.3f}"
        )
        rel = tt - tt[0]
        prof = []
        for k in range(0, int(dur), 2):
            w = (rel >= k) & (rel < k + 2)
            if w.any():
                prof.append(f"{q['cmd_x_vel'][w].mean():+.2f}/{q['cmd_theta_vel'][w].mean():+.2f}")
        print("   2초별 명령 x/θ:", " ".join(prof[:30]) + (" …" if len(prof) > 30 else ""))


def train_reference(repo: str) -> None:
    root = f"{CACHE}/{repo}"
    info_path = f"{root}/meta/info.json"
    if not os.path.exists(info_path):
        print(f"학습 {repo}: 로컬 캐시에 없음 — 학습 대조 생략 (내려받지 않음)")
        return
    info = json.load(open(info_path))
    fps = info["fps"]
    names = (info["features"].get("action") or {}).get("names")
    if isinstance(names, list) and names[-2:] != ["x.vel", "theta.vel"]:
        print(f"학습 {repo}: action 끝 두 칸이 {names[-2:]} — 베이스가 뒤에 있지 않다. 적분을 믿지 말 것")
    files = sorted(glob.glob(f"{root}/data/*/*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in files])
    a = np.stack(d["action"].to_numpy())
    if a.shape[1] != 16:
        print(f"학습 {repo}: action 이 {a.shape[1]}D — 베이스 칸(14·15)이 없어 학습 대조 생략")
        return
    ep = d["episode_index"].to_numpy()
    th, xs, dur = [], [], []
    for e in np.unique(ep):
        m = ep == e
        th.append(np.degrees(a[m, TH].sum() / fps))
        xs.append(a[m, X].sum() / fps)
        dur.append(m.sum() / fps)
    th, xs, dur = map(np.array, (th, xs, dur))
    k = fps / 21.0
    print(
        f"학습 {repo}: {len(th)}ep, 표기 {fps} fps | 명령 적분 회전 중앙 {np.median(th):+.0f}° "
        f"(p10 {np.percentile(th, 10):+.0f} / p90 {np.percentile(th, 90):+.0f}) | "
        f"전진 중앙 {np.median(xs):+.2f} m (p10 {np.percentile(xs, 10):+.2f} / p90 {np.percentile(xs, 90):+.2f}) | "
        f"길이 중앙 {np.median(dur):.1f}s"
    )
    print(
        f"   실제 녹화가 21 Hz 였다면 (×{k:.2f}): 회전 {np.median(th) * k:+.0f}° · 전진 {np.median(xs) * k:+.2f} m · "
        f"길이 {np.median(dur) * k:.1f}s — 어느 쪽인지는 녹화 로그의 Control loop rate 로 확인"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--train", help="학습 데이터셋 repo id (로컬 캐시만)")
    args = ap.parse_args()
    if args.train:
        try:
            train_reference(args.train)
        except Exception as e:  # 학습 대조가 죽어도 eval 구간은 낸다
            print(f"학습 {args.train}: 대조 실패 ({type(e).__name__}: {e}) — 생략")
    for run in args.runs:
        run_segments(run)


if __name__ == "__main__":
    sys.exit(main())
