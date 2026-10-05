#!/usr/bin/env python
"""실기 eval 기록에서 지연 셋을 잰다 (로봇에 연결하지 않는다 -- 남은 파일만 읽는다).

    uv run --no-sync python scripts/eval_latency_stats.py <회차> [<회차> ...]
    uv run --no-sync python scripts/eval_latency_stats.py 1002_1150_m1_t04_e30 --train kiroaiseoul/task04_pour_liquid_from_tubes_to_beaker

<회차> 는 eval_najy.sh 가 찍는 이름(``MMDD_HHMM_<모델>_t<단계>_e<exec>``). 회차마다 읽는 것:
  ~/eval_logs/<회차>.basevel.csv          베이스 명령·실측 속도 (LEROBOT_BASE_VEL_LOG)
  ~/.cache/huggingface/lerobot/kiroaiseoul/eval_najy_<회차>   eval 데이터셋 (팔 action·state)
  ~/eval_logs/<회차>.log                   루프 요약줄 (Control loop rate ... per-frame)

내는 것:
  베이스 지연   cmd[t] 와 meas[t+L] 의 오차가 최소인 L (틱) -- 순수 지연에 응답 시간이 더해진 「유효 지연」.
               명령 계단에서 잰 t63·t90 도 낸다. 같은 행의 meas 는 그 틱 *맨 앞*에서 읽고 cmd 는
               *맨 끝*에 쓰므로 cmd[t] 가 처음 보이는 것은 meas[t+1] 이다 -- L=1 이 「한 틱 안에 반응」.
               초 단위는 실제 틱 간격(t_mono 차분 중앙값)으로 환산한다. 정지 명령 뒤 잔여 회전도 낸다.
  팔 지연       action[t] 와 observation.state[t+L] 의 팔 12관절 MAE 가 최소인 L (틱). ``--train`` 을 주면
               같은 계산을 학습 데이터에도 해서 나란히 낸다 (학습 데이터는 2틱이었다).
  추론 스파이크 policy 구간 루프 요약줄의 (1/min − 1/mean) -- exec 마다 한 번 몰리는 추론이 만드는 최장 틱의 초과분.
               `other=` 는 30프레임 평균이라 추론 시간을 1/exec 로 희석해서 보여 준다 (참고로만 낸다).
"""

import argparse
import glob
import re
from pathlib import Path

import numpy as np

LOGS = Path.home() / "eval_logs"
CACHE = Path.home() / ".cache/huggingface/lerobot"
ARM = [i for i in range(14) if i not in (6, 13)]
MAX_LAG = 10


def best_lag(cmd, meas, lags, min_active=20):
    """cmd[t] 와 meas[t+L] 의 평균 절대오차가 최소인 L. cmd 가 움직이는 구간만 센다."""
    out = {}
    for L in lags:
        c, m = cmd[: len(cmd) - L], meas[L:]
        if len(c) == 0:
            continue
        act = np.abs(c).sum(-1) > 1e-3 if c.ndim > 1 else np.abs(c) > 1e-3
        if act.sum() < min_active:
            continue
        out[L] = float(np.abs(c[act] - m[act]).mean())
    if not out:
        return None, out
    return min(out, key=out.get), out


def base_stats(run):
    f = LOGS / f"{run}.basevel.csv"
    if not f.exists():
        return f"베이스: {f} 없음"
    import pandas as pd

    df = pd.read_csv(f)
    df = df[df["phase"] == "policy"]
    if len(df) < 50:
        return f"베이스: policy 행 {len(df)}개 -- 너무 적다"
    t = df["t_mono"].to_numpy()
    # 에피소드 경계(1초 넘는 간격)로 끊어 따로 맞춘 뒤 합친다
    cuts = np.nonzero(np.diff(t) > 1.0)[0] + 1
    lines = []
    dt = np.median(np.diff(t)[np.diff(t) < 1.0])
    for name, c, m in (
        ("회전", "cmd_theta_vel", "meas_theta_vel"),
        ("전진", "cmd_x_vel", "meas_x_vel"),
    ):
        errs = {}
        for seg in np.split(np.arange(len(df)), cuts):
            if len(seg) < 30:
                continue
            _, e = best_lag(
                df[c].to_numpy()[seg],
                df[m].to_numpy()[seg],
                range(0, MAX_LAG + 1),
                min_active=5,
            )
            for L, v in e.items():
                errs.setdefault(L, []).append(v)
        if not errs:
            lines.append(f"  {name}: 명령이 거의 0 -- 지연을 잴 움직임이 없다")
            continue
        mean = {L: float(np.mean(v)) for L, v in errs.items()}
        L = min(mean, key=mean.get)
        lines.append(
            f"  {name}: 지연 {L} 틱 ≈ {L * dt * 1e3:.0f} ms (틱 {dt * 1e3:.1f} ms) | "
            + " ".join(f"L{k}={v:.3f}" for k, v in sorted(mean.items()))
        )
    th_c, th_m = df["cmd_theta_vel"].to_numpy(), df["meas_theta_vel"].to_numpy()
    # 계단 응답 t63/t90: 명령이 0.1 rad/s 넘게 바뀌고 10틱 이상 그대로인 지점에서, 실측이 변화량의 63%/90% 에
    # 처음 닿는 틱 (measure_base.py latency 와 같은 정의, 단 정책이 낸 계단이라 크기·방향이 제각각이다)
    t63, t90 = [], []
    for s in range(1, len(th_c) - 10):
        step = th_c[s] - th_c[s - 1]
        if (
            abs(step) < 0.1
            or np.any(np.abs(np.diff(th_c[s : s + 10])) > 1e-6)
            or np.any(np.diff(t[s - 1 : s + 10]) > 1.0)
        ):
            continue
        base0 = th_m[s - 1] if s > 0 else 0.0
        prog = (th_m[s : s + 40] - base0) / step
        hit63, hit90 = np.nonzero(prog >= 0.63)[0], np.nonzero(prog >= 0.90)[0]
        if len(hit63):
            t63.append(hit63[0])
        if len(hit90):
            t90.append(hit90[0])
    if t63:
        lines.append(
            f"  회전 계단 응답: t63 중앙값 {np.median(t63):.0f} 틱 ≈ {np.median(t63) * dt * 1e3:.0f} ms"
            + (
                f", t90 {np.median(t90):.0f} 틱 ≈ {np.median(t90) * dt * 1e3:.0f} ms"
                if t90
                else ""
            )
            + f" ({len(t63)}개 계단; k 틱 = 명령을 쓴 뒤 k 번째 틱 머리에서 읽은 실측)"
        )
    # 정지 명령 뒤 잔여 회전: 명령 |theta| 가 0.02 를 넘다가 0 이 된 지점 이후 실측 적분 (도)
    stops = np.nonzero((np.abs(th_c[:-1]) > 0.02) & (np.abs(th_c[1:]) < 1e-3))[0] + 1
    resid = []
    for s in stops:
        e = s
        while e < len(th_c) and abs(th_c[e]) < 1e-3 and e - s < int(2.0 / dt):
            e += 1
        resid.append(np.degrees(np.abs(th_m[s:e]).sum() * dt))
    if resid:
        lines.append(
            f"  정지 뒤 잔여 회전: 중앙값 {np.median(resid):.2f}°, 최대 {np.max(resid):.2f}° ({len(resid)}회)"
        )
    return "베이스 (policy 구간):\n" + "\n".join(lines)


def load_ds(repo_dir):
    import pandas as pd

    files = sorted(glob.glob(str(repo_dir / "data/**/*.parquet"), recursive=True))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(x) for x in files])
    return [
        (
            ep,
            np.stack(g["action"].values)[:, :14],
            np.stack(g["observation.state"].values)[:, :14],
        )
        for ep, g in df.groupby("episode_index")
    ]


def arm_lag(eps):
    errs = {}
    for _, a, s in eps:
        _, e = best_lag(a[:, ARM], s[:, ARM], range(0, MAX_LAG + 1))
        for L, v in e.items():
            errs.setdefault(L, []).append(v)
    if not errs:
        return None, {}
    mean = {L: float(np.mean(v)) for L, v in errs.items()}
    return min(mean, key=mean.get), mean


def arm_stats(run, train):
    eps = load_ds(CACHE / "kiroaiseoul" / f"eval_najy_{run}")
    if eps is None:
        return f"팔: eval 데이터셋 eval_najy_{run} 이 로컬에 없다"
    L, mean = arm_lag(eps)
    out = [
        f"팔 (eval {len(eps)}ep): 지연 {L} 틱 | "
        + " ".join(f"L{k}={v:.4f}" for k, v in sorted(mean.items()))
    ]
    if train:
        teps = load_ds(CACHE / train)
        if teps is None:
            out.append(f"  학습 {train}: 로컬에 없다")
        else:
            tL, tmean = arm_lag(teps)
            out.append(
                f"  학습 {train} ({len(teps)}ep): 지연 {tL} 틱 | "
                + " ".join(f"L{k}={v:.4f}" for k, v in sorted(tmean.items()))
            )
    return "\n".join(out)


def infer_stats(run):
    f = LOGS / f"{run}.log"
    if not f.exists():
        return f"추론: {f} 없음"
    pat = re.compile(
        r"Control loop rate .*phase=policy.*mean=([\d.]+) Hz, min=([\d.]+) Hz.*?(?:other=(\d+)ms)?$"
    )
    spikes, others = [], []
    for line in f.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            mean_hz, min_hz = float(m.group(1)), float(m.group(2))
            spikes.append((1 / min_hz - 1 / mean_hz) * 1e3)
            if m.group(3):
                others.append(int(m.group(3)))
    if not spikes:
        return "추론: policy 구간 루프 요약줄이 없다"
    return (
        f"추론 스파이크 (최장 틱 − 평균 틱): 중앙값 {np.median(spikes):.0f} ms ({len(spikes)}창)"
        + (
            f" | other= 중앙값 {np.median(others):.0f} ms (30프레임 평균, 추론이 희석됨)"
            if others
            else ""
        )
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+")
    ap.add_argument(
        "--train", default=None, help="학습 데이터셋 repo_id (팔 지연을 나란히 비교)"
    )
    a = ap.parse_args()
    for run in a.runs:
        print(f"== {run}")
        for part in (base_stats(run), arm_stats(run, a.train), infer_stats(run)):
            print(part)


if __name__ == "__main__":
    main()
