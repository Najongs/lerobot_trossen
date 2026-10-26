"""가장 최근 test_ep1 record·replay 베이스 로그를 비교한다 (scripts/test_ep1.sh compare 가 부른다).

  uv run python scripts/test_ep1_compare.py ~/eval_logs/test_ep1/kiroaiseoul_TEST_EP1

로그는 LEROBOT_BASE_VEL_LOG(base_vel_log.py) 형식: 한 줄 = 제어 루프 한 틱.
- 실제 루프 속도: t_mono 간격의 중앙값 → Hz. 녹화가 21 fps 표기보다 느리게 돌았다면
  21 fps 재생은 그만큼 빨라지고 베이스 이동이 줄어든다.
- 이동량: 실측 속도(meas_*)를 실제 시간(t_mono)으로 적분한 전진 m · 회전 °.
  녹화는 손으로 민 실제 움직임, 재생은 그 명령을 받은 베이스의 실제 움직임이다.
- 재생의 명령/실측 비: 베이스가 명령을 얼마나 따라갔나 (1.0 이면 그대로).

녹화 로그엔 리셋 구간(teleop)도 섞일 수 있다 — 에피소드 1개면 lerobot 이 마지막 리셋을 건너뛰어 거의 없다.
"""

import csv
import glob
import math
import statistics
import sys


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return [{k: (float(v) if k != "phase" else v) for k, v in r.items()} for r in rows]


def summarize(rows):
    t = [r["t_mono"] for r in rows]
    dt = [b - a for a, b in zip(t, t[1:])]
    out = {"ticks": len(rows), "dur": t[-1] - t[0] if len(t) > 1 else 0.0}
    out["hz"] = 1 / statistics.median(dt) if dt else float("nan")
    out["long_ticks"] = sum(d > 0.1 for d in dt)
    for key in ("meas", "cmd"):
        x = sum(r[f"{key}_x_vel"] * d for r, d in zip(rows, dt))
        th = sum(r[f"{key}_theta_vel"] * d for r, d in zip(rows, dt))
        out[key] = (x, math.degrees(th))
    return out


def latest(prefix, mode):
    files = sorted(glob.glob(f"{prefix}_{mode}_*.csv"))
    return files[-1] if files else None


def main():
    prefix = sys.argv[1]
    paths = {m: latest(prefix, m) for m in ("record", "replay")}
    stats = {}
    for mode, path in paths.items():
        if path is None:
            print(f"{mode}: 로그 없음 ({prefix}_{mode}_*.csv) — 그 단계를 test_ep1.sh 로 다시 돌린다")
            continue
        s = summarize(load(path))
        stats[mode] = s
        print(f"{mode}: {path}")
        print(
            f"  {s['ticks']} 틱 · {s['dur']:.1f} s · 루프 {s['hz']:.1f} Hz (0.1 s 넘는 틱 {s['long_ticks']})"
        )
        print(f"  실측 이동 전진 {s['meas'][0]:+.3f} m · 회전 {s['meas'][1]:+.1f}°")
        if mode == "replay":
            print(f"  명령 적분 전진 {s['cmd'][0]:+.3f} m · 회전 {s['cmd'][1]:+.1f}°")
    if len(stats) < 2:
        return
    rec, rep = stats["record"], stats["replay"]

    def ratio(a, b):
        return f"{a / b:.2f}" if abs(b) > 1e-3 else "—"

    print("비교 (재생 − 녹화):")
    print(f"  루프 속도 {ratio(rep['hz'], rec['hz'])} 배 · 길이 {ratio(rep['dur'], rec['dur'])} 배")
    # 왕복 경로면 끝 순이동이 0 에 가까워 비율이 터진다(10/26 TEST_EP2 「회전 12 배」 = 실제 4.5°) — 차이로 본다
    print(
        f"  끝 위치 차: 전진 {rep['meas'][0] - rec['meas'][0]:+.3f} m · 방향 {rep['meas'][1] - rec['meas'][1]:+.1f}°"
        "  (0 이면 같은 자리·같은 방향, 바퀴 오도메트리 기준)"
    )
    print(
        "  재생 명령 − 녹화 실측: 회전 "
        f"{rep['cmd'][1] - rec['meas'][1]:+.1f}°  (방향 보정이 더한 몫 — 끔이면 ≈0)"
    )


if __name__ == "__main__":
    main()
