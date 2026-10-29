#!/usr/bin/env python
"""베이스를 cam_high 로 닫아 직진한다 — 달리는 동안 방향(yaw)을 잡고, 거리는 카메라가 잰 이동량으로 멈춘다.

왜: 직진 오도는 ≈2% 과대(비 0.978 로 보정 가능)지만 달리는 중 방향이 틀어지는 것(10/26 TEST_EP2: 4°)은 오도가 모른다.
base_turn.py 와 같은 구조 — 출발 프레임이 기준(0°, 0 m), depth_yaw.motion 으로 yaw·앞쪽 이동(수평)을 ≈3 Hz 로 잰다.
  방향: ω = clip(Kp·(0 − yaw), ±max) — 움직이는 중이라 작은 각속도도 먹는다(정지 중 1~2° 스텝은 유격 안에서 논다)
  거리: 카메라 이동량(depth 비율 오차 ×0.945 보정)이 목표 − 리드에 닿으면 정지 → 재측정 → |오차| > tol 이면 저속 재접근(≤3 패스)
  기준 프레임은 0.4 m 마다 멈춰 갱신(겹침 유지). --stop-at 을 주면 화면 가운데 가장 가까운 depth 가 그 거리면 멈춘다.

  uv run python scripts/base_drive.py --m 0.5             # 🚨 베이스가 앞으로 0.5 m (음수면 뒤로)
  uv run python scripts/base_drive.py --m 0.5 --then -0.5 # 왕복 — 처음 프레임 대비 끝 위치·방향으로 검산
  uv run python scripts/base_drive.py --m 1.0 --stop-at 0.6   # 앞 물체 0.6 m 에서 멈춤 (거리 목표와 둘 중 먼저)
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_approach import capture, clearance, open_cam  # noqa: E402
from depth_probe import SERIALS  # noqa: E402
from base_turn import CameraTurner  # noqa: E402
from depth_yaw import motion  # noqa: E402
from measure_base import _check_alive, _connect, _wrap  # noqa: E402

DEPTH_SCALE_FIX = 0.945  # D405 cam_high depth 가 거리를 ≈6% 크게 잰다 (docs/depth_cam_high_pc2_1027.md, 줄자 3회)


class CameraDriver:
    def __init__(self, base, cam, scale, K, speed=0.1, creep=0.03, lead_m=0.02, tol_m=0.01, passes=3,
                 chunk_m=0.4, kp=0.5, max_rate_deg=3.0, yaw_deadband=0.5, stop_at=None, end_turn=True, ramp_s=0.0, log=print):
        self.base, self.cam, self.scale, self.K = base, cam, scale, K
        self.speed, self.creep, self.lead, self.tol, self.passes = speed, creep, lead_m, tol_m, passes
        self.ramp_s = ramp_s  # 출발 가속 시간 — 10/28 개방루프에서 출발 1 s 동안 ω 가 튀며 몸체가 +2.7° 돌았다
        self.chunk, self.kp, self.max_rate, self.deadband = chunk_m, kp, math.radians(max_rate_deg), yaw_deadband
        self.stop_at, self.log, self.end_turn = stop_at, log, end_turn
        self.overshoots = []
        self.turner = CameraTurner(base, cam, scale, K, log=log)  # 도착 뒤 남은 방향 오차를 닫는다 (검증된 회전 모듈)

    def _frame(self, n=1):
        if n > 1:
            color, depth = capture(self.cam, n, self.scale)
            return cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), depth
        self.cam.async_read(timeout_ms=500)
        with self.cam.frame_lock:
            return (cv2.cvtColor(self.cam.latest_color_frame, cv2.COLOR_RGB2GRAY),
                    self.cam.latest_depth_frame.astype(np.float32) * self.scale)

    def _odom(self):
        self.base.update_state()
        x, y, th = self.base.get_pose()
        return x, y, th

    def _pose(self, gray, depth, anchors, near, fast):
        """anchors [(fwd_m, yaw, frame)] 중 near 에 가장 가까운 것 기준 → (앞쪽 누적 m, yaw °, 인라이어)."""
        af, ay, ref = min(anchors, key=lambda a: abs(a[0] - near))
        r = motion(*ref, gray, depth, self.K, iters=200 if fast else 2000, nfeatures=1500 if fast else 4000)
        if r["yaw_deg"] is None:
            return None, None, 0
        fwd = af + r["fwd_m"] * DEPTH_SCALE_FIX
        yaw = ay + r["yaw_deg"]
        if not fast:
            r2 = motion(gray, depth, *ref, self.K)
            if r2["yaw_deg"] is not None:
                fwd = af + (r["fwd_m"] - r2["fwd_m"]) / 2 * DEPTH_SCALE_FIX
                yaw = ay + (r["yaw_deg"] - r2["yaw_deg"]) / 2
        return fwd, yaw, r["inliers"]

    def _settled(self, anchors, near, n=5):
        g, d = self._frame(15)
        fs, ys, inls = [], [], []
        for _ in range(n):
            gi, _ = self._frame()
            f, y, inl = self._pose(gi, d, anchors, near, fast=False)
            if f is not None:
                fs.append(f)
                ys.append(y)
                inls.append(inl)
        if not fs:
            return None, None, 0, (g, d)
        return float(np.median(fs)), float(np.median(ys)), int(np.median(inls)), (g, d)

    def _drive_until(self, anchors, target, lead, speed):
        """카메라를 읽으며 직진(방향 유지)하다 target − lead 에서 정지. 궤적 [(t, fwd, yaw, 인라이어, clearance)]."""
        traj, misses, stopped_by = [], 0, None
        g, d = self._frame(5)
        f, y, _ = self._pose(g, d, anchors, target, fast=True)
        if f is None:
            return None, None
        sign = 1.0 if target > f else -1.0
        t0 = time.perf_counter()
        ok = self.base.set_cmd_vel(sign * (speed if self.ramp_s <= 0 else 0.2 * speed), 0.0)
        if ok is False:
            self.log("    set_cmd_vel 이 False 를 돌려줬다")
        moving_since = None  # 무반응 감시: 명령 중 실측 속도가 0 이면
        try:
            while True:
                g, d = self._frame()
                f, y, inl = self._pose(g, d, anchors, target, fast=True)
                now = time.perf_counter() - t0
                self.base.update_state()
                if abs(self.base.get_vel()[0]) < 0.1 * speed:
                    moving_since = moving_since or now
                    if now - moving_since > 1.5:
                        self.log(f"    베이스 무반응 — 명령 {sign * speed:+.2f} m/s 인데 실측 속도 0 이 1.5 s 넘음. 구간 중단")
                        stopped_by = "no_response"
                        break
                else:
                    moving_since = None
                if f is None:
                    misses += 1
                    if misses > 10:
                        self.log("    카메라 측정 연속 실패 — 멈춘다")
                        stopped_by = "vo_fail"
                        break
                    continue
                misses = 0
                clr = clearance(d) * DEPTH_SCALE_FIX if sign > 0 else float("nan")
                traj.append((now, f, y, inl, clr, math.degrees(self.base.get_pose()[2]), *self.base.get_vel()[:2]))
                remaining = (target - f) * sign
                if remaining <= lead:
                    break
                if self.stop_at is not None and sign > 0 and clr <= self.stop_at:
                    stopped_by = "clearance"
                    break
                # 방향 유지: 출발 방향(0°)으로. **후진 중엔 끈다** — 10/28 `013213` 후진 0.2 m 에서 최대 12° 흔들리다 +7.9° 로 끝났다
                # (캐스터가 앞서 끌려 불안정 [추정]). 전진만 잡는다
                w = 0.0 if (sign < 0 or abs(y) < self.deadband) else max(-self.max_rate, min(self.max_rate, math.radians(-self.kp * y)))
                v = self.creep if remaining <= 0.06 else speed
                if self.ramp_s > 0:
                    v = min(v, max(0.2, now / self.ramp_s) * speed)
                self.base.set_cmd_vel(sign * v, w)
                if now > abs(target - traj[0][1]) / self.creep + 10:
                    self.log("    시간 초과 — 멈춘다")
                    stopped_by = "timeout"
                    break
        finally:
            self.base.set_cmd_vel(0.0, 0.0)
        time.sleep(0.8)
        return traj, stopped_by

    def drive(self, m: float) -> dict:
        """상대 m(+ 앞)만큼 직진. 출발 프레임이 (0 m, 0°). |m| > chunk 면 구간을 나눠 구간 끝마다 기준 갱신."""
        g0, d0 = self._frame(15)
        anchors = [(0.0, 0.0, (g0, d0))]
        x0, y0, th0 = self._odom()
        n_chunks = max(1, math.ceil(abs(m) / self.chunk))
        hist, stopped_by = [], None
        for c in range(1, n_chunks + 1):
            sub = m * c / n_chunks
            lead = float(np.median(self.overshoots)) if self.overshoots else self.lead
            traj, stopped_by = self._drive_until(anchors, sub, lead, self.speed)
            if traj is None:
                raise RuntimeError("VO 를 못 잰다 — cam_high 에 텍스처가 안 보인다")
            f, y, inl, (g, d) = self._settled(anchors, sub)
            hist.append({"chunk": c, "sub_target": sub, "stop_cmd_at": traj[-1][1] if traj else None, "settled_m": f,
                         "yaw_deg": y, "inliers": inl, "samples": len(traj), "stopped_by": stopped_by,
                         "yaw_max_abs_during": max((abs(t[2]) for t in traj), default=None),
                         "odom_th_settled": math.degrees(self._odom()[2]),
                         # (t s, 카메라 앞 m, 카메라 yaw °, 오도 θ °, 실측 v, 실측 ω) — 방향이 언제 틀어지는지 보려고
                         "traj": _traj_json(traj)})
            if traj and f is not None:
                ov = (f - traj[-1][1]) * (1.0 if m > 0 else -1.0)
                if 0 <= ov <= 0.1:
                    self.overshoots.append(ov)
            if f is not None:
                anchors.append((f, y, (g, d)))
            if stopped_by == "clearance":
                break
        n = 0
        while stopped_by != "clearance" and f is not None and abs(m - f) > self.tol and n < self.passes:
            n += 1
            t2, _ = self._drive_until(anchors, m, 0.005, self.creep)
            f, y, inl, (g, d) = self._settled(anchors, m)
            hist.append({"pass": n, "stop_cmd_at": t2[-1][1] if t2 else None, "settled_m": f, "yaw_deg": y, "inliers": inl,
                         "odom_th_settled": math.degrees(self._odom()[2]), "traj": _traj_json(t2 or [])})
        turn_fix = None
        if self.end_turn and y is not None and abs(y) > 0.5 and stopped_by != "no_response":
            # 남은 방향 오차는 회전 모듈로 닫는다 — 달리면서 세게 잡으면 지연(VO 0.3 s + 베이스 0.3 s) 때문에 발산한다(10/28 `013436`: 게인 2.5 → 7°)
            self.log(f"    도착 방향 {y:+.2f}° → 회전 모듈로 되돌린다")
            turn_fix = self.turner.turn(-y)
            f2, y2, inl2, (g, d) = self._settled(anchors, m)
            hist.append({"end_turn": -y, "turn_err": turn_fix["err"], "settled_m": f2, "yaw_deg": y2, "inliers": inl2})
            if f2 is not None:
                f, y = f2, y2
        x1, y1, th1 = self._odom()
        odom_fwd = (x1 - x0) * math.cos(th0) + (y1 - y0) * math.sin(th0)
        res = {"target_m": m, "final_m": f, "err_m": None if f is None else m - f, "final_yaw_deg": y, "passes": n,
               "end_turn_deg": None if turn_fix is None else turn_fix["target"],
               "odom_fwd_m": odom_fwd, "odom_yaw_deg": math.degrees(_wrap(th1 - th0)), "stopped_by": stopped_by,
               "hist": hist, "last_frame": (g, d)}
        self.log(f"    목표 {m:+.3f} m → 카메라 {f:+.3f} m (오차 {res['err_m'] * 100:+.1f} cm, 방향 {y:+.2f}°, 패스 {n}"
                 f"{', ' + stopped_by if stopped_by else ''}) · 오도 {odom_fwd:+.3f} m / {res['odom_yaw_deg']:+.2f}° · 기준 {len(anchors)}개")
        return res


def _traj_json(traj):
    return [(round(t[0], 2), round(t[1], 4), round(t[2], 2), round(t[5], 2), round(t[6], 3), round(t[7], 3)) for t in traj]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, required=True, help="상대 거리 m (+ 앞)")
    ap.add_argument("--then", type=float, action="append", default=[])
    ap.add_argument("--pc", default=os.environ.get("TROSSEN_PC", "pc1"), choices=list(SERIALS))
    ap.add_argument("--speed", type=float, default=0.1)
    ap.add_argument("--stop-at", type=float, default=None, help="앞 장애물(화면 가운데 1 퍼센타일 depth) 이 거리면 멈춤 (m)")
    ap.add_argument("--no-heading", action="store_true", help="방향 유지 끔 (비교용)")
    ap.add_argument("--kp", type=float, default=0.5, help="주행 중 방향 유지 게인 (°/s per °) — 약하게. 남은 오차는 도착 뒤 회전 모듈이 닫는다")
    ap.add_argument("--max-rate", type=float, default=3.0, help="주행 중 방향 유지 각속도 상한 °/s")
    ap.add_argument("--ramp", type=float, default=0.0, help="출발 가속 시간 s (0 = 계단)")
    ap.add_argument("--no-end-turn", action="store_true", help="도착 뒤 방향 되돌리기 끔")
    ap.add_argument("--max-m", type=float, default=0.2, help="이동 절댓값 상한 (10/28 사용자 지정 0.2 — 넓히려면 명시)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--out", default="outputs/base_drive")
    args = ap.parse_args()
    plan = [args.m] + args.then
    if any(abs(x) > args.max_m for x in plan):
        raise SystemExit(f"|거리| ≤ {args.max_m}")
    out = Path(args.out) / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dry" if args.dry_run else ""))
    out.mkdir(parents=True, exist_ok=True)
    cam, scale, K = open_cam(args.pc, 640, 480, 30)
    base, cleanup = _connect(args.dry_run)
    results = []
    try:
        if not args.dry_run:
            _check_alive(base, 0, probe=0.05)
        drv = CameraDriver(base, cam, scale, K, speed=args.speed, stop_at=args.stop_at, kp=0.0 if args.no_heading else args.kp,
                           max_rate_deg=args.max_rate, end_turn=not args.no_end_turn, ramp_s=args.ramp)
        g_first, d_first = drv._frame(15)
        for i, m in enumerate(plan):
            if not args.yes:
                input(f"\n  [{i + 1}/{len(plan)}] {m:+.2f} m 직진 — 진행 방향 비었으면 Enter: ")
            r = drv.drive(m)
            g, d = r.pop("last_frame")
            cv2.imwrite(str(out / f"drive{i}_color.png"), g)
            np.save(out / f"drive{i}_depth_m.npy", d)
            results.append(r)
        g, d = drv._frame(15)
        net = motion(g_first, d_first, g, d, K)
        import marker_yaw as mk
        up = mk.floor_up(d_first, K)
        m0, m1 = mk.measure(g_first, d_first, up), mk.measure(g, d, up)
        mk_net = [f"{k.split(':')[-1]} {-mk.wrap(m1[k]['az_deg'] - m0[k]['az_deg'], 360):+.2f}°" for k in m0 if k in m1 and (":flat" in k or k.startswith("pair"))]
        print(f"  마커 방향 변화(처음→끝): {' · '.join(mk_net) or '없음'}")
        print(f"\n  계획 합 {sum(plan):+.2f} m · 처음 프레임 대비 끝: 앞 {net['fwd_m'] * DEPTH_SCALE_FIX:+.3f} m · 옆 {net['left_m'] * DEPTH_SCALE_FIX:+.3f} m "
              f"· 방향 {net['yaw_deg']:+.2f}° (인라이어 {net['inliers']}) · 오도 합 {sum(r['odom_fwd_m'] for r in results):+.3f} m")
    finally:
        cleanup()
        cam.disconnect()
    (out / "report.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2, ensure_ascii=False, default=float))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
