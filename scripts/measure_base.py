#!/usr/bin/env python
"""베이스 계측 — 정책 부정확의 원인 중 하드웨어에 속한 부분을 실측한다.

오프라인 데이터로는 답이 안 나온 것들이 있다. 기록된 parquet 의 ``timestamp`` 는
lerobot 이 명목 fps 로 만들어 낸 **합성값**이라(dt std = 0.00ms) 취득 시 실제 제어율을
알 수 없고, 속도 명령에 베이스가 얼마나 늦게 반응하는지도 데이터에 남지 않는다.
이 스크립트는 그 둘을 로봇에서 직접 잰다.

왜 중요한가: 팔은 절대 관절각이라 명령이 늦어도 목표에 도달할 뿐이지만, 베이스는
속도라서 ``위치 = 속도 x Δt`` 다. 제어 주기가 곧 적분 구간이므로 추론이 느려진 만큼
**그대로 과회전한다.** 취득 21.5Hz 에서 배운 명령을 평가 5.7Hz 로 실행하면 3.8배 돈다.

세 가지를 잰다::

    latency   set_cmd_vel() 계단 입력 후 get_vel() 이 목표의 63%/90% 에 닿는 시간.
              Mobile ALOHA 의 BASE_DELAY=13 (260ms @50Hz) 에 해당하는 우리 값.
    odom      get_pose() 로 90도 회전을 반복해 드리프트를 본다. 오도메트리를 관측에
              넣을지 판단하는 전제 -- 드리프트가 크면 넣어도 소용없다.
    rate      set_cmd_vel/get_vel 왕복에 드는 시간. 제어 루프의 베이스 몫 하한선.

사용법::

    cd ~/lerobot_trossen
    uv run --extra pi0 python scripts/measure_base.py latency
    uv run --extra pi0 python scripts/measure_base.py odom --turns 6
    uv run --extra pi0 python scripts/measure_base.py rate

전부 베이스만 쓴다 (팔 IP 불필요). ``--dry-run`` 은 로봇 없이 흐름만 확인한다.

🚨 latency 와 odom 은 **베이스를 실제로 움직인다.** 진행 방향을 비우고 비상정지에
   손이 닿는 위치에서 실행할 것. 비상정지가 걸려 있으면 모터 토크가 안 걸려 측정이
   조용히 0 만 읽는다 -- 스크립트가 그 상태를 감지해 중단한다.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _connect(dry_run: bool):
    """베이스만 연 뒤 (base, 정리함수) 를 준다. 팔은 이 측정에 필요 없다."""
    if dry_run:
        class _Fake:
            """0.25s 시상수 1차 지연 + 약간의 오도메트리 드리프트를 흉내낸다."""
            def __init__(self):
                self._v = [0.0, 0.0]
                self._cmd = [0.0, 0.0]
                self._pose = [0.0, 0.0, 0.0]
                self._t = time.perf_counter()

            def _step(self):
                now = time.perf_counter()
                dt = now - self._t
                self._t = now
                a = 1.0 - math.exp(-dt / 0.25)
                for i in range(2):
                    self._v[i] += (self._cmd[i] - self._v[i]) * a
                self._pose[0] += self._v[0] * dt * math.cos(self._pose[2])
                self._pose[2] += self._v[1] * dt * 1.02      # 2% 스케일 오차
                return dt

            def update_state(self): self._step(); return True
            def get_vel(self): self._step(); return list(self._v)
            def get_pose(self): self._step(); return list(self._pose)
            def set_cmd_vel(self, x, t): self._step(); self._cmd = [x, t]; return True
            def enable_motor_torque(self, on): return True
            def init_base(self): return True
        return _Fake(), (lambda: None)

    from trossen_slate import TrossenSlate

    base = TrossenSlate()
    base.init_base()
    base.enable_motor_torque(True)

    def cleanup():
        try:
            base.set_cmd_vel(0.0, 0.0)
        finally:
            base.enable_motor_torque(False)

    return base, cleanup


def _check_alive(base, axis: int, probe: float = 0.25, secs: float = 0.6) -> None:
    """모터가 실제로 도는지 확인한다.

    비상정지가 걸려 있거나 토크가 안 걸리면 set_cmd_vel 은 성공을 반환하면서
    get_vel 은 계속 0 을 준다. 그 상태로 측정하면 "지연 무한대" 라는 무의미한
    숫자가 나오므로 먼저 걸러 낸다.
    """
    cmd = [0.0, 0.0]
    cmd[axis] = probe
    base.set_cmd_vel(*cmd)
    peak = 0.0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        base.update_state()
        peak = max(peak, abs(base.get_vel()[axis]))
        time.sleep(0.01)
    base.set_cmd_vel(0.0, 0.0)
    time.sleep(0.3)
    if peak < probe * 0.15:
        raise SystemExit(
            f"베이스가 명령에 반응하지 않는다 (명령 {probe}, 관측 최대 {peak:.4f}).\n"
            "  비상정지 버튼이 눌려 있는지, 베이스 전원이 들어와 있는지 확인할 것."
        )


def measure_latency(base, axis: int, cmd: float, hold: float, rest: float, reps: int,
                    poll: float = 0.005):
    """계단 입력에 대한 상승 시간. 각 반복의 (t63, t90, 정상상태) 를 모은다."""
    label = "x.vel (전진)" if axis == 0 else "theta.vel (회전)"
    print(f"\n=== 지연 측정: {label}, 명령 {cmd}, {reps}회 ===")
    out = []
    for r in range(reps):
        base.set_cmd_vel(0.0, 0.0)
        time.sleep(rest)

        samples = []
        c = [0.0, 0.0]
        c[axis] = cmd
        t0 = time.perf_counter()
        base.set_cmd_vel(*c)
        # 폴링 간격을 고정한다. 자유 루프로 돌리면 드라이버 캐시가 갱신되기 전에
        # 같은 값을 여러 번 읽어 샘플만 부풀고, 실기에서는 Modbus 왕복이 간격을
        # 대신 정해 버려 측정 해상도가 장비마다 달라진다.
        nxt = t0
        while True:
            el = time.perf_counter() - t0
            if el > hold:
                break
            base.update_state()
            samples.append((time.perf_counter() - t0, base.get_vel()[axis]))
            nxt += poll
            time.sleep(max(0.0, nxt - time.perf_counter()))
        base.set_cmd_vel(0.0, 0.0)

        # 정상상태는 후반 30% 의 평균으로 잡는다 (초반 과도응답 제외).
        tail = [v for t, v in samples if t > hold * 0.7]
        ss = sum(tail) / len(tail) if tail else 0.0
        if abs(ss) < abs(cmd) * 0.2:
            print(f"  [{r+1}] 정상상태 {ss:+.4f} 가 명령 {cmd} 에 비해 너무 작다 -- 건너뜀")
            continue

        def first_at(frac):
            thr = ss * frac
            for t, v in samples:
                if (v >= thr) if ss > 0 else (v <= thr):
                    return t
            return None

        t63, t90 = first_at(0.63), first_at(0.90)
        out.append({"t63": t63, "t90": t90, "steady": ss, "n": len(samples)})
        print(f"  [{r+1}] 정상상태 {ss:+.4f}  t63 {_ms(t63)}  t90 {_ms(t90)}  샘플 {len(samples)}")
        time.sleep(rest)

    if not out:
        print("  유효한 반복이 없다.")
        return {"axis": label, "reps": []}

    t63s = [o["t63"] for o in out if o["t63"] is not None]
    t90s = [o["t90"] for o in out if o["t90"] is not None]
    ss = [o["steady"] for o in out]
    print(f"\n  t63 중앙값 {_ms(_med(t63s))}   t90 중앙값 {_ms(_med(t90s))}")
    print(f"  정상상태 평균 {sum(ss)/len(ss):+.4f} (명령 {cmd}, 비율 {sum(ss)/len(ss)/cmd:.3f})")
    if t63s:
        for fps in (30, 20, 10):
            print(f"    -> {fps}Hz 에서 BASE_DELAY 상당 = {_med(t63s) * fps:.1f} 스텝")
    return {"axis": label, "cmd": cmd,
            "t63_median": _med(t63s), "t90_median": _med(t90s),
            "steady_ratio": sum(ss) / len(ss) / cmd, "reps": out}


def measure_odom(base, turns: int, cmd: float, target_deg: float):
    """제자리 회전을 반복하며 get_pose() 의 theta 가 얼마나 정확한지 본다.

    각 회전마다 pose 를 열어 두고, 목표 각도에 닿으면 멈춘다. 누적 오차가 회전 수에
    비례해 커지면 드리프트, 일정 비율로 어긋나면 스케일 오차다 -- 후자는 보정 가능하다.
    """
    print(f"\n=== 오도메트리: {target_deg}도 회전 {turns}회, 명령 {cmd} rad/s ===")
    base.update_state()
    start = base.get_pose()
    print(f"  시작 pose  x={start[0]:+.4f} y={start[1]:+.4f} th={math.degrees(start[2]):+.2f}도")

    rows = []
    tgt = math.radians(target_deg)
    for i in range(turns):
        base.update_state()
        th0 = base.get_pose()[2]
        sign = 1.0 if (i % 2 == 0) else -1.0        # 왕복시켜 제자리로 돌아오게 한다
        t0 = time.perf_counter()
        base.set_cmd_vel(0.0, cmd * sign)
        moved = 0.0
        while abs(moved) < tgt:
            if time.perf_counter() - t0 > tgt / max(abs(cmd), 1e-6) * 3 + 5:
                print("  시간 초과 -- 중단")
                break
            base.update_state()
            moved = _wrap(base.get_pose()[2] - th0)
            time.sleep(0.005)
        base.set_cmd_vel(0.0, 0.0)
        time.sleep(0.8)                              # 관성으로 더 도는 분까지 포함해 읽는다
        base.update_state()
        final = _wrap(base.get_pose()[2] - th0)
        over = math.degrees(abs(final)) - target_deg
        rows.append({"i": i, "sign": sign, "reported_deg": math.degrees(final), "overshoot_deg": over})
        print(f"  [{i+1}] 방향 {'+' if sign > 0 else '-'}  "
              f"pose 보고 {math.degrees(final):+7.2f}도  (목표 {target_deg}, 초과 {over:+.2f})")

    base.update_state()
    end = base.get_pose()
    dx, dy = end[0] - start[0], end[1] - start[1]
    dth = math.degrees(_wrap(end[2] - start[2]))
    print(f"\n  종료 pose  x={end[0]:+.4f} y={end[1]:+.4f} th={math.degrees(end[2]):+.2f}도")
    print(f"  제자리 왕복 후 누적 변화: 위치 {math.hypot(dx, dy):.4f} m, 각도 {dth:+.2f}도")
    print("  (왕복이 짝수면 이상적으로 0. 위치 변화는 회전 중심이 틀어진 양이다.)")
    print("\n  ** 이 값은 pose 가 스스로 보고한 숫자다. 실제 각도와 맞는지는")
    print("     바닥에 표시를 하고 눈으로 확인해야 한다 -- 오도메트리 자체가 틀릴 수 있다. **")
    return {"turns": rows, "net_xy_m": math.hypot(dx, dy), "net_theta_deg": dth}


def measure_rate(base, n: int):
    """베이스 왕복(set_cmd_vel + update_state + get_vel) 에 드는 시간.

    제어 루프에서 베이스가 차지하는 몫의 하한선이다. 이 값이 크면 카메라와 팔을
    아무리 줄여도 목표 fps 에 못 간다.
    """
    print(f"\n=== 베이스 I/O 주기: {n}회 ===")
    base.set_cmd_vel(0.0, 0.0)
    rd, wr = [], []
    for _ in range(n):
        t = time.perf_counter(); base.update_state(); base.get_vel()
        rd.append(time.perf_counter() - t)
        t = time.perf_counter(); base.set_cmd_vel(0.0, 0.0)
        wr.append(time.perf_counter() - t)
    tot = [a + b for a, b in zip(rd, wr)]
    for lbl, v in (("읽기", rd), ("쓰기", wr), ("합계", tot)):
        print(f"  {lbl}  평균 {_ms(sum(v)/len(v))}  중앙 {_ms(_med(v))}  최악 {_ms(max(v))}")
    print(f"\n  베이스만으로 가능한 상한 주파수 약 {1/(sum(tot)/len(tot)):.0f} Hz")
    return {"read_ms": sum(rd)/len(rd)*1e3, "write_ms": sum(wr)/len(wr)*1e3,
            "total_ms": sum(tot)/len(tot)*1e3}


def _med(v):
    if not v:
        return None
    s = sorted(v)
    return s[len(s)//2] if len(s) % 2 else (s[len(s)//2 - 1] + s[len(s)//2]) / 2


def _ms(x):
    return "n/a" if x is None else f"{x*1e3:6.1f}ms"


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["latency", "odom", "rate", "all"])
    ap.add_argument("--dry-run", action="store_true", help="로봇 없이 흐름만 확인")
    ap.add_argument("--cmd-linear", type=float, default=0.15, help="전진 계단 입력 (m/s)")
    ap.add_argument("--cmd-angular", type=float, default=0.5, help="회전 계단 입력 (rad/s)")
    ap.add_argument("--hold", type=float, default=1.5, help="계단 유지 시간 (s)")
    ap.add_argument("--rest", type=float, default=1.0, help="반복 사이 정지 시간 (s)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--poll", type=float, default=0.005,
                    help="지연 측정 샘플 간격 (s). 실기 Modbus 왕복보다 짧으면 의미 없다")
    ap.add_argument("--turns", type=int, default=4, help="odom: 회전 횟수 (짝수 권장)")
    ap.add_argument("--turn-deg", type=float, default=90.0)
    ap.add_argument("--samples", type=int, default=200, help="rate: 반복 횟수")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    if not args.dry_run and args.mode in ("latency", "odom", "all"):
        print("🚨 베이스가 실제로 움직인다. 진행 방향을 비우고 비상정지에 손이 닿는 위치에서.")
        print("   중단하려면 Ctrl-C (스크립트가 정지 명령을 보낸다).")
        input("   준비되면 Enter: ")

    base, cleanup = _connect(args.dry_run)
    res = {"mode": args.mode, "dry_run": args.dry_run}
    try:
        if args.mode in ("latency", "all"):
            _check_alive(base, 1)
            res["latency_angular"] = measure_latency(
                base, 1, args.cmd_angular, args.hold, args.rest, args.reps, args.poll)
            res["latency_linear"] = measure_latency(
                base, 0, args.cmd_linear, args.hold, args.rest, args.reps, args.poll)
        if args.mode in ("odom", "all"):
            _check_alive(base, 1)
            res["odom"] = measure_odom(base, args.turns, args.cmd_angular, args.turn_deg)
        if args.mode in ("rate", "all"):
            res["rate"] = measure_rate(base, args.samples)
    except KeyboardInterrupt:
        print("\n중단됨 -- 베이스를 정지시킨다.")
    finally:
        cleanup()

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n-> {args.out}")

    if args.mode in ("latency", "all") and res.get("latency_angular", {}).get("t63_median"):
        t63 = res["latency_angular"]["t63_median"]
        print("\n" + "=" * 70)
        print("다음 단계: 이 t63 을 취득 제어율과 묶어 본다.")
        print(f"  회전축 t63 = {t63*1e3:.0f}ms")
        print("  취득 로그에서:  grep 'Control loop rate' <취득로그> | grep phase=policy | tail -1")
        print("  30Hz 취득이었다면 BASE_DELAY 상당 = %.1f 스텝." % (t63 * 30))
        print("  평가 제어율이 취득보다 낮으면 그 비율만큼 과회전한다 (속도 x Δt).")
        print("=" * 70)


if __name__ == "__main__":
    main()
