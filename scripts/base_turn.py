#!/usr/bin/env python
"""베이스를 상대 각도만큼 cam_high 로 닫아 돈다 — 오도메트리 대신 카메라(VO)가 「얼마나 돌았나」를 잰다.

왜: 바퀴 오도메트리는 직전 동작(캐스터 자세)에 따라 같은 명령이 −2~+6° 다르게 돌고 방향 전환 때 지연·초과가 들쭉날쭉하다
(docs/depth_cam_high_pc2_1027.md). 카메라로 닫으면 ±0.2° (10/28 실측, 마커 없이 VO 만으로).

방법: 돌면서 cam_high 를 ≈3 Hz 로 읽어 기준 프레임 대비 yaw(depth_yaw.motion, 바닥 평면 기준)를 재고 목표 − 리드에서 멈춘다 →
0.8 s 뒤 재측정 → |오차| > tol 이면 4°/s 로 다시 다가간다(최대 3 패스). 큰 회전은 ≤30° 구간으로 쪼개 구간 끝마다 멈춰
깨끗한(15 프레임 중앙값 depth) 기준 프레임을 잡고 각도를 누적한다 — 돌면서 흐릿한 단일 프레임으로 기준을 갈아타면 60° 에 ±2° 흔들렸다(10/28 `005101`). 작은 보정을 오도 양으로 명령하지 않는 것이 핵심 — 정지 뒤 1~2° 스텝은 유격 안에서 논다.

  uv run python scripts/base_turn.py --deg -80            # 🚨 베이스가 돈다. 상대 각도 (+ 왼쪽)
  uv run python scripts/base_turn.py --deg 60 --then -60  # 돌았다 되돌아와 왕복 오차를 본다
라이브러리: CameraTurner(base, cam, scale, K).turn(deg) → dict(final_yaw, err, passes, …)
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
from depth_approach import capture, open_cam  # noqa: E402
from depth_probe import SERIALS  # noqa: E402
from depth_yaw import motion  # noqa: E402
from measure_base import _check_alive, _connect, _wrap  # noqa: E402


def _marker_az(gray, depth):
    """진단: 눕힌 마커 PnP 방위(°) — VO 판정을 같은 프레임의 외부 기준과 대조 (10/28 오후 VO 가 마커와 5~19° 어긋남). 안 보이면 None."""
    try:
        import marker_yaw as mk
        m = mk.measure(gray, depth, mk.floor_up(depth))
        return next((v["az_deg"] for k, v in m.items() if k.endswith(":flat")), None)
    except Exception:
        return None


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class CameraTurner:
    def __init__(self, base, cam, scale, K, speed_deg=10.0, creep_deg=4.0, lead_deg=2.5, tol_deg=0.5,
                 passes=3, chunk_deg=15.0, stop_by="vo", min_move_deg=0.0, backoff_deg=10.0, log=print):
        self.base, self.cam, self.scale, self.K = base, cam, scale, K
        self.speed, self.creep, self.lead, self.tol = speed_deg, creep_deg, lead_deg, tol_deg
        self.passes, self.chunk, self.log = passes, chunk_deg, log
        self.overshoots = []
        # "odom": 한 동작 안의 정지 시점을 바퀴 회전량으로 — 10°/s 안팎에선 정지 후 몸체 회전 ≈ 바퀴 회전(10/28 `141823` 0.9~1.07)인데
        # 도는 중 몸체(VO)는 바퀴보다 3~8° 뒤처져 VO 로 멈추면 매번 넘어간다. 동작 사이 보정은 그대로 정지 VO
        self.stop_by = stop_by
        # 남은 각이 min_move 보다 작으면 반대로 backoff 만큼 물러났다가 같은 방향·정속으로 다가간다 — 3~5° 를 직접 돌면 저속 감김–풀림으로 2.5~3° 남았다(10/28)
        self.min_move, self.backoff = min_move_deg, backoff_deg
        # odom 정지의 몸체/바퀴 비 — 방향별로 동작마다 정착 VO 로 갱신 (10/28 `144837`: 이 자리에서 0.45~0.75 라 1.0 가정이 매번 3° 모자랐다)
        self.ratio, self._move = {1.0: 1.0, -1.0: 1.0}, None

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
        return math.degrees(self.base.get_pose()[2])

    def _yaw(self, gray, depth, anchors, near, fast):
        """anchors [(yaw, frame)] 중 near 에 가장 가까운 것 기준으로 잰 절대 yaw (시작 = 0).
        전체 품질이면 양방향(기준→현재, 현재→기준) 평균 — depth 노이즈가 양쪽에 다르게 들어가 일부 상쇄."""
        ay, ref = min(anchors, key=lambda a: abs(a[0] - near))
        r = motion(*ref, gray, depth, self.K, iters=200 if fast else 2000, nfeatures=1500 if fast else 4000)
        if r["yaw_deg"] is None:
            return None, 0
        if fast:
            return ay + r["yaw_deg"], r["inliers"]
        r2 = motion(gray, depth, *ref, self.K)
        if r2["yaw_deg"] is None:
            return ay + r["yaw_deg"], r["inliers"]
        return ay + (r["yaw_deg"] - r2["yaw_deg"]) / 2, min(r["inliers"], r2["inliers"])

    def _yaw_settled(self, anchors, near, n=5):
        """정지 중 n 프레임 각각 전체 품질 VO → 중앙값. (yaw, 인라이어 중앙값, 마지막 (gray, depth 중앙값))."""
        ys, inls = [], []
        g, d = self._frame(15)  # depth 는 15 프레임 중앙값
        for _ in range(n):
            gi, _ = self._frame()
            y, inl = self._yaw(gi, d, anchors, near, fast=False)
            if y is not None:
                ys.append(y)
                inls.append(inl)
        self.last_marker = _marker_az(g, d)
        if not ys:
            return None, 0, (g, d)
        return float(np.median(ys)), int(np.median(inls)), (g, d)

    def _drive_until(self, anchors, target, lead, speed_deg, y0=None):
        """카메라를 읽으며 돌다 target − lead 에서 정지. 돌면서 anchor_step 마다 기준 프레임 추가. 궤적을 준다."""
        traj, misses = [], 0
        # y0 = 직전 정착 측정 — odom 정지의 기준점이라 빠른 VO 한 번(수 ° 흔들림)으로 잡으면 그만큼 틀린다(10/28 `145249` 바퀴 28° 과주행)
        y = y0
        if y is None:
            g, d = self._frame(5)
            y, _ = self._yaw(g, d, anchors, target, fast=True)
        if y is None:
            y, _ = self._yaw(g, d, anchors, target, fast=False)  # 빠른 모드(1500 특징)가 텍스처 적은 장면에서 실패하면 전체 품질로
        if y is None:
            return None
        sign = 1.0 if target > y else -1.0
        y_start, o_start = y, self._odom()
        t0 = time.perf_counter()
        self.base.set_cmd_vel(0.0, sign * math.radians(speed_deg))
        try:
            while True:
                g, d = self._frame()
                y, inl = self._yaw(g, d, anchors, target, fast=True)
                if y is None:
                    y, inl = self._yaw(g, d, anchors, target, fast=False)
                now = time.perf_counter() - t0
                if y is None:
                    misses += 1
                    if misses > 10:
                        self.log("    카메라 측정 연속 실패 — 멈춘다")
                        break
                    continue
                misses = 0
                o = self._odom()
                traj.append((now, y, inl, o))
                y_stop = y if self.stop_by == "vo" else y_start + self.ratio[sign] * _wrap_deg(o - o_start)
                remaining = (target - y_stop) * sign
                if remaining <= lead:
                    break
                self.base.set_cmd_vel(0.0, sign * math.radians(self.creep if remaining <= 6.0 else speed_deg))
                if now > abs(target - traj[0][1]) / self.creep + 10:
                    self.log("    시간 초과 — 멈춘다")
                    break
        finally:
            self.base.set_cmd_vel(0.0, 0.0)
        time.sleep(0.8)
        self._move = (sign, y_start, o_start)
        return traj

    def _learn(self, y_settled):
        """방금 동작의 정착 몸체 회전 ÷ 바퀴 회전으로 그 방향 비를 갱신."""
        if self._move is None or y_settled is None:
            return
        sign, y0, o0 = self._move
        do = _wrap_deg(self._odom() - o0)
        self._move = None
        if abs(do) > 2.0 and do * sign > 0:
            r = (y_settled - y0) / do
            if 0.2 <= r <= 1.6:
                self.ratio[sign] = 0.5 * self.ratio[sign] + 0.5 * r

    def turn(self, deg: float) -> dict:
        """상대 deg(+ 왼쪽)만큼 돈다. 시작 프레임이 0°. |deg| > chunk 면 구간을 나눠 구간 끝마다 멈춰 기준을 잡는다."""
        g0, d0 = self._frame(10)
        anchors = [(0.0, (g0, d0))]
        odom0 = self._odom()
        marker0 = _marker_az(g0, d0)
        n_chunks = max(1, math.ceil(abs(deg) / self.chunk))
        hist, samples, y = [], 0, 0.0
        if self.min_move and abs(deg) < self.min_move:
            n_chunks = 0  # 작은 회전은 처음부터 물러났다 다가가기 (아래 패스 루프)
        for c in range(1, n_chunks + 1):
            sub = deg * c / n_chunks
            lead = (float(np.median(self.overshoots)) if self.overshoots else self.lead) if self.stop_by == "vo" else 0.5
            traj = self._drive_until(anchors, sub, lead, self.speed, y0=y)
            if traj is None:
                raise RuntimeError("VO 를 못 잰다 — cam_high 에 텍스처가 안 보인다")
            samples += len(traj)
            y, inl, (g, d) = self._yaw_settled(anchors, sub)
            self._learn(y)
            hist.append({"chunk": c, "sub_target": sub, "stop_cmd_at": traj[-1][1] if traj else None, "settled": y, "inliers": inl, "marker_az": self.last_marker,
                         "traj": [(round(t, 2), round(a, 2), round(o, 2)) for t, a, _, o in traj]})
            if traj and y is not None and self.stop_by == "vo":
                ov = (y - traj[-1][1]) * (1.0 if deg > 0 else -1.0)
                if 0 <= ov <= 8:
                    self.overshoots.append(ov)
            if y is not None:
                anchors.append((y, (g, d)))  # 멈춘 자리의 깨끗한 프레임이 다음 구간·재접근 패스의 기준 (겹침 최대)
        traj = [(0, 0, 0, 0)] * samples
        n = 0
        while y is not None and abs(deg - y) > self.tol and n < self.passes:
            n += 1
            if self.min_move and abs(deg - y) < self.min_move:
                back = deg - math.copysign(self.backoff, deg - y)
                self._drive_until(anchors, back, 0.5 if self.stop_by == "odom" else self.lead, self.speed, y0=y)
                yb, _, (g, d) = self._yaw_settled(anchors, back)
                self._learn(yb)
                if yb is not None:
                    anchors.append((yb, (g, d)))
                    y = yb
                hist.append({"backoff_to": back, "settled": yb, "marker_az": self.last_marker})
                t2 = self._drive_until(anchors, deg, 0.5 if self.stop_by == "odom" else self.lead, self.speed, y0=y)
            else:
                t2 = self._drive_until(anchors, deg, 0.5, self.creep, y0=y)
            y, inl, (g, d) = self._yaw_settled(anchors, deg)
            self._learn(y)
            hist.append({"pass": n, "stop_cmd_at": t2[-1][1] if t2 else None, "settled": y, "inliers": inl, "marker_az": self.last_marker,
                         "traj": [(round(t, 2), round(a, 2), round(o, 2)) for t, a, _, o in (t2 or [])]})
        res = {"target": deg, "final_yaw": y, "err": None if y is None else deg - y, "passes": n,
               "odom_delta": math.degrees(_wrap(math.radians(self._odom() - odom0))), "hist": hist,
               "anchors": [a[0] for a in anchors], "marker_az0": marker0, "ratio": dict(self.ratio), "samples": len(traj), "last_frame": (g, d)}
        self.log(f"    목표 {deg:+.1f}° → 카메라 {y:+.2f}° (오차 {res['err']:+.2f}, 패스 {n}, 오도 {res['odom_delta']:+.2f}, "
                 f"기준 {len(anchors)}개, 샘플 {len(traj)})")
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=float, required=True, help="상대 회전 (°, + 왼쪽)")
    ap.add_argument("--then", type=float, action="append", default=[], help="이어서 돌 각도 (반복 가능)")
    ap.add_argument("--pc", default=os.environ.get("TROSSEN_PC", "pc1"), choices=list(SERIALS))
    ap.add_argument("--creep", type=float, default=4.0, help="목표 6° 안·재접근 회전 속도 °/s (저속은 바퀴만 돌고 몸체가 멈췄다 튄다, 10/28)")
    ap.add_argument("--stop-by", choices=["vo", "odom"], default="vo", help="한 동작 안의 정지 판정 (odom: 바퀴 회전량, 동작 사이는 카메라)")
    ap.add_argument("--min-move", type=float, default=0.0, help="남은 각이 이보다 작으면 물러났다 다가간다 (0 = 끔)")
    ap.add_argument("--backoff", type=float, default=10.0)
    ap.add_argument("--max-deg", type=float, default=90.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--out", default="outputs/base_turn")
    args = ap.parse_args()
    plan = [args.deg] + args.then
    if any(abs(x) > args.max_deg for x in plan):
        raise SystemExit(f"|각도| ≤ {args.max_deg}")
    out = Path(args.out) / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dry" if args.dry_run else ""))
    out.mkdir(parents=True, exist_ok=True)
    cam, scale, K = open_cam(args.pc, 640, 480, 30)
    base, cleanup = _connect(args.dry_run)
    results = []
    try:
        if not args.dry_run:
            _check_alive(base, 1, probe=0.15)
        turner = CameraTurner(base, cam, scale, K, creep_deg=args.creep, stop_by=args.stop_by, min_move_deg=args.min_move, backoff_deg=args.backoff)
        g_first, d_first = turner._frame(15)
        cv2.imwrite(str(out / "first_color.png"), g_first)
        np.save(out / "first_depth_m.npy", d_first)
        for i, deg in enumerate(plan):
            if not args.yes:
                input(f"\n  [{i + 1}/{len(plan)}] {deg:+.1f}° 회전 — 주변(팔 끝 원) 비었으면 Enter: ")
            r = turner.turn(deg)
            g, d = r.pop("last_frame")
            cv2.imwrite(str(out / f"turn{i}_color.png"), g)
            np.save(out / f"turn{i}_depth_m.npy", d)
            results.append(r)
        # 왕복 검산: 처음 프레임 대비 끝 프레임 (계획 합이 0 이면 0 이어야)
        g, d = turner._frame(15)
        net = motion(g_first, d_first, g, d, K)
        net2 = motion(g, d, g_first, d_first, K)
        import marker_yaw as mk
        up = mk.floor_up(d_first, K)
        m0, m1 = mk.measure(g_first, d_first, up), mk.measure(g, d, up)
        mk_net = [f"{k.split(':')[-1]} {-mk.wrap(m1[k]['az_deg'] - m0[k]['az_deg'], 360):+.2f}" for k in m0 if k in m1 and (":flat" in k or k.startswith("pair"))]
        print(f"\n  계획 합 {sum(plan):+.1f}° · 처음 프레임 대비 끝 VO {net['yaw_deg'] if net['yaw_deg'] is None else round(net['yaw_deg'], 2)}° / 역방향 "
              f"{net2['yaw_deg'] if net2['yaw_deg'] is None else round(-net2['yaw_deg'], 2)}° (인라이어 {net['inliers']}) · 마커 {' · '.join(mk_net) or '없음'} "
              f"· 오도 합 {sum(r['odom_delta'] for r in results):+.2f}°")
    finally:
        cleanup()
        cam.disconnect()
    (out / "report.json").write_text(json.dumps({"plan": plan, "results": results}, indent=2, ensure_ascii=False, default=float))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
