#!/usr/bin/env python
"""베이스 제자리 회전을 cam_high depth 로 재서 오도메트리와 대조한다 — 각도기 대신.

순서(오도 yaw 누적 목표): 0 → +15 → +30 → 0 → −15 → −30 → 0 (°, + = 왼쪽/CCW). 정지마다 15 프레임 중앙값 depth.
회전각은 scripts/depth_yaw.py (RGB-D 비주얼 오도메트리, 바닥 평면 기준 yaw) — 회전 중심이 어디든, depth 비율 오차와도 무관.

  uv run python scripts/depth_rotate.py --dry-run --yes    # 베이스 가짜(무접촉), 카메라는 실제 — 흐름 확인
  uv run python scripts/depth_rotate.py                    # 🚨 베이스가 실제로 제자리에서 돈다

🚨 dry-run 이 아니면 베이스가 제자리에서 최대 ±--max-deg(기본 30°) 돈다. 매 회전 전 Enter.
   주변(특히 팔 끝이 그리는 원)을 비우고 비상정지에 손이 닿는 곳에서. 시작 확인 동작으로 ≈5° 먼저 돈다.
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from depth_approach import capture, open_cam  # noqa: E402
from depth_probe import SERIALS, colorbar, colorize  # noqa: E402
from depth_yaw import motion  # noqa: E402
from measure_base import _check_alive, _connect, _drive_to, _wrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="베이스를 가짜로 (카메라는 실제로 연다)")
    ap.add_argument("--pc", default=os.environ.get("TROSSEN_PC", "pc1"), choices=list(SERIALS))
    ap.add_argument("--max-deg", type=float, default=30.0, help="회전 절댓값 상한 (°)")
    ap.add_argument("--plan", default="15,30,0,-15,-30,0", help="오도 yaw 누적 목표 (°, 쉼표)")
    ap.add_argument("--speed-deg", type=float, default=15.0, help="회전 속도 °/s (데드밴드 3°/s 보다 충분히 크게)")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--reversal-comp", default="0,0",
                    help="방향 전환 때 오도 목표에 더 줄 각도 °: 「왼→오른,오른→왼」 (10/27 실측 지연 4.3,1.1)")
    ap.add_argument("--caster-reset", choices=["none", "fwd", "shuffle"], default="none",
                    help="회전 직전 보조(캐스터) 바퀴 정렬: fwd = 앞 --reset-m (런 끝에 되돌림) · shuffle = 앞 뒤 (제자리). "
                         "10/27 사용자 힌트 — 캐스터가 처음 어떤 자세냐에 따라 회전이 되고 안 되고 한다")
    ap.add_argument("--reset-m", type=float, default=0.1)
    ap.add_argument("--feedback", choices=["odom", "marker", "vo", "live"], default="odom",
                    help="marker/vo: 오도로 간 뒤 카메라로 정지 0 대비 실제 각도를 재서 남은 오차만큼 다시 돈다. "
                         "live: 돌면서 카메라(마커, 없으면 VO)를 계속 읽어 목표에 닿는 순간 멈춘다 — 정지 뒤 작은 보정 스텝이 유격 안에서 노는 문제를 피한다")
    ap.add_argument("--live-lead", type=float, default=2.5,
                    help="live: 목표 몇 ° 앞에서 정지 명령 (관성·지연 몫) 초기값 — 이후 관측된 초과량 중앙값으로 자동 갱신 (10/28 실측 2.5~2.8)")
    ap.add_argument("--vo-only", action="store_true", help="live/marker: 마커가 보여도 무시하고 VO 만 쓴다 (마커 없는 현장 흉내)")
    ap.add_argument("--center-marker", action="store_true",
                    help="시작 때 눕힌 마커가 화면 가운데 오도록 베이스를 돌린 뒤 기준을 잡는다 — ±30° 내내 마커가 보이게")
    ap.add_argument("--live-speed-deg", type=float, default=10.0, help="live: 회전 속도, 목표 6° 안에서는 4°/s")
    ap.add_argument("--live-passes", type=int, default=3,
                    help="live: 멈춘 뒤 |오차| > tol 이면 4°/s · 리드 0.5° 로 다시 다가가는 추가 패스 수 (전환 뒤 초과량이 0.3~4° 로 들쭉날쭉해서)")
    ap.add_argument("--feedback-tol", type=float, default=0.5)
    ap.add_argument("--feedback-iters", type=int, default=4)
    ap.add_argument("--yes", action="store_true", help="Enter 확인 생략 (dry-run 에서만)")
    ap.add_argument("--out", default="outputs/depth_rotate")
    args = ap.parse_args()

    plan = [float(v) for v in args.plan.split(",") if v.strip()]
    comp_l2r, comp_r2l = (float(v) for v in args.reversal_comp.split(","))
    if any(abs(t) > args.max_deg + 1e-9 for t in plan):
        raise SystemExit(f"--plan 에 ±{args.max_deg}° 를 넘는 목표가 있다")
    if args.yes and not args.dry_run:
        raise SystemExit("--yes 는 dry-run 에서만")

    out = Path(args.out) / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_dry" if args.dry_run else ""))
    out.mkdir(parents=True, exist_ok=True)
    cam, scale, K = open_cam(args.pc, 640, 480, 30)
    base, cleanup = _connect(args.dry_run)
    stops, frames = [], []
    try:
        base.update_state()
        x0, y0, th0 = base.get_pose()  # 오도 0 점 = 시작 확인 동작 전
        if not args.dry_run:
            _check_alive(base, 1, probe=0.15)  # 0.15 rad/s · 0.6 s — ≈5° 돌았다 멈춘다

        def pose():
            base.update_state()
            x, y, th = base.get_pose()
            return math.degrees(_wrap(th - th0)), math.hypot(x - x0, y - y0)

        def measure(k: int, label: str):
            color, depth = capture(cam, args.frames, scale)
            yaw, xy = pose()
            cv2.imwrite(str(out / f"stop{k}_color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            vis = colorize(depth, 0.5, 2.5, 0.5)
            cv2.imwrite(str(out / f"stop{k}_depth.png"), np.hstack([vis, colorbar(vis.shape[0], 0.5, 2.5, 0.5)]))
            np.save(out / f"stop{k}_depth_m.npy", depth)
            frames.append((cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), depth))
            r = motion(*frames[0], *frames[-1], K) if k else {"yaw_deg": 0.0, "inliers": None}
            stops.append({"stop": k, "label": label, "odom_yaw_deg": yaw, "odom_xy_drift_m": xy,
                          "depth_yaw_from0_deg": r["yaw_deg"], "vo_inliers": r["inliers"]})
            vo = "매칭 부족" if r["yaw_deg"] is None else f"{r['yaw_deg']:+7.2f}° (인라이어 {r['inliers']})"
            print(f"[정지 {k} {label}] 오도 yaw {yaw:+7.2f}° · 위치 흔들림 {xy * 100:.1f} cm"
                  + (f" · depth VO yaw(정지0 대비) {vo}" if k else ""))

        import marker_yaw as mk
        if args.center_marker and not args.dry_run:
            for _ in range(3):
                color, depth = capture(cam, 5, scale)
                dets = [c for n, i, c in mk.detect(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY))]
                if not dets:
                    print("    마커 정렬: 마커가 안 보인다 — 건너뛴다")
                    break
                # 눕힌 마커 = 화면에서 더 아래쪽(바닥)에 있는 것
                c = max(dets, key=lambda c: c.mean(0)[1])
                dx = c.mean(0)[0] - 320
                ang = math.degrees(math.atan(dx / K[0]))  # 오른쪽(+dx)이면 오른쪽(−)으로 돈다
                print(f"    마커 정렬: 화면 x {c.mean(0)[0]:.0f} → {-ang:+.1f}° 돈다")
                if abs(ang) < 2.0:
                    break
                _drive_to(base, 1, math.radians(-ang), math.radians(args.speed_deg), math.radians(5), 10)
            base.update_state()
            x0, y0, th0 = base.get_pose()  # 정렬 뒤를 오도 0 점으로
        measure(0, "시작")
        ref_up = mk.floor_up(frames[0][1], K)
        ref_m = mk.measure(frames[0][0], frames[0][1], ref_up) if args.feedback in ("marker", "live") and not args.vo_only else {}
        if ref_m:
            # 이 런에서 보이는 사전만 뒤진다 — 8개 다 돌리면 검출이 ≈0.3 s 라 live 가 느리다
            mk.DICTS[:] = sorted({k.split(":")[0] for k in ref_m if k.startswith("DICT")}) or mk.DICTS
            print(f"    기준 마커: {', '.join(k for k in ref_m)} · 사전 {mk.DICTS}")

        anchors = [(0.0, frames[0])]  # VO 기준 프레임 후보: (그 정지의 몸체 yaw, (gray, depth)). 정지마다 추가

        def cam_yaw(gray, depth, fast: bool = False, near: float | None = None):
            """정지 0 대비 실제 yaw (°, + 왼쪽) — 눕힌 마커 > 두 마커 중심선 > VO.
            VO 는 near(예상 각)에 가장 가까운 정지를 기준으로 재서 겹침을 키운다 — 정지 0 하나만 쓰면 33° 에서 −2° 치우쳤다(10/28 `004230`)."""
            if args.feedback in ("marker", "live"):
                m = mk.measure(gray, depth, ref_up)
                for pref in (":flat", "pair"):
                    for k in m:
                        if pref in k and k in ref_m:
                            return -mk.wrap(m[k]["az_deg"] - ref_m[k]["az_deg"], 360), k.split(":")[-1]
            base_yaw, ref = min(anchors, key=lambda a: abs(a[0] - (near if near is not None else 0.0)))
            r = motion(*ref, gray, depth, K, iters=200 if fast else 2000, nfeatures=1500 if fast else 4000)
            return (None if r["yaw_deg"] is None else base_yaw + r["yaw_deg"]), ("vo" if base_yaw == 0.0 else f"vo@{base_yaw:+.0f}")

        def live_turn(body_tgt: float, slow: bool = False, lead: float | None = None):
            """카메라를 읽으며 돌다가 목표 −lead 에서 멈춘다. 궤적 [(t, cam_yaw, src, odom_yaw)] 을 준다."""
            traj, misses = [], 0
            wl = math.radians(4.0 if slow else args.live_speed_deg)
            lead_use = lead_now() if lead is None else lead
            color, depth = capture(cam, 5, scale)
            y, src = cam_yaw(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), depth, fast=True, near=body_tgt)
            if y is None:
                print("    live: 시작 각도를 못 잰다 — 오도로 간다")
                return None
            sign = 1.0 if body_tgt > y else -1.0
            t0 = time.perf_counter()
            base.set_cmd_vel(0.0, sign * wl)
            try:
                while True:
                    cam.async_read(timeout_ms=500)
                    with cam.frame_lock:
                        g = cv2.cvtColor(cam.latest_color_frame, cv2.COLOR_RGB2GRAY)
                        d = cam.latest_depth_frame.astype(np.float32) * scale
                    y, src = cam_yaw(g, d, fast=True, near=body_tgt)
                    now = time.perf_counter() - t0
                    if y is None:
                        misses += 1
                        if misses > 15:
                            print("    live: 카메라 측정 연속 실패 — 멈춘다")
                            break
                        continue
                    misses = 0
                    traj.append((now, y, src, pose()[0]))
                    remaining = (body_tgt - y) * sign
                    if remaining <= lead_use:
                        break
                    creep = remaining <= 6.0 or src == "vo"  # VO 는 ≈3 Hz 라 빠르게 돌면 한 샘플에 3° 지나간다
                    base.set_cmd_vel(0.0, sign * (math.radians(4.0) if creep else wl))
                    if now > abs(body_tgt - traj[0][1]) / args.live_speed_deg * 3 + 8:
                        print("    live: 시간 초과 — 멈춘다")
                        break
            finally:
                base.set_cmd_vel(0.0, 0.0)
            time.sleep(0.8)
            return traj

        w = math.radians(args.speed_deg)
        fwd_total = 0.0
        overshoots = []  # live: 정지 명령 시점 각 → 멈춘 뒤 각 (같은 방향 양수)

        def lead_now():
            return float(np.median(overshoots)) if overshoots else args.live_lead

        def caster_reset():
            nonlocal fwd_total
            if args.caster_reset == "none":
                return
            fwd_total += _drive_to(base, 0, args.reset_m, 0.08, 0.03, 10)
            if args.caster_reset == "shuffle":
                fwd_total += _drive_to(base, 0, -args.reset_m, 0.08, 0.03, 10)

        # 의도한 몸체 각도는 계획대로(시작 확인 동작 전 원점 기준), 오도 목표에는 전환 보정을 누적해 얹는다
        last_sign = 1.0 if not args.dry_run else 0.0  # 시작 확인 동작은 왼쪽(+) 회전
        prev_tgt, odom_goal = 0.0, 0.0
        for i, tgt in enumerate(plan, 1):
            intended = tgt - prev_tgt if i > 1 else tgt - stops[0]["odom_yaw_deg"]
            body_tgt = tgt - stops[0]["odom_yaw_deg"]  # 정지 0 대비 의도 몸체 각
            sign = 1.0 if intended > 0 else -1.0
            comp = 0.0
            if last_sign and sign != last_sign:
                comp = (comp_l2r if sign < 0 else comp_r2l) * sign
            odom_goal = (odom_goal if i > 1 else stops[0]["odom_yaw_deg"]) + intended + comp
            cur, _ = pose()
            delta = odom_goal - cur
            if not args.yes:
                input(f"\n  {delta:+.1f}° 회전 (의도 {intended:+.1f}°, 전환 보정 {comp:+.1f}°"
                      f"{', 캐스터 정렬 ' + args.caster_reset if args.caster_reset != 'none' else ''}) — 주변 비었으면 Enter: ")
            caster_reset()
            fb, traj = [], None
            if args.feedback == "live":
                traj = live_turn(body_tgt)
            if traj is None:
                _drive_to(base, 1, math.radians(delta), w, math.radians(5), abs(math.radians(delta)) / w * 3 + 5)
            if traj is not None:
                color, depth = capture(cam, args.frames, scale)
                y, src = cam_yaw(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), depth, near=body_tgt)
                fb.append({"cam_yaw": y, "src": src, "err": None if y is None else body_tgt - y, "odom_yaw": pose()[0],
                           "live_samples": len(traj), "live_last": traj[-1][1] if traj else None,
                           "live_hz": len(traj) / traj[-1][0] if len(traj) > 1 and traj[-1][0] > 0 else None,
                           "lead_used": lead_now()})
                if traj and y is not None and traj[-1][2] != "vo":
                    ov = (y - traj[-1][1]) * (1.0 if body_tgt > traj[0][1] else -1.0)
                    if 0 <= ov <= 8:
                        overshoots.append(ov)
                passes = 0
                while y is not None and abs(body_tgt - y) > args.feedback_tol and passes < args.live_passes:
                    passes += 1
                    t2 = live_turn(body_tgt, slow=True, lead=0.5)
                    if t2 is None:
                        break
                    color, depth = capture(cam, args.frames, scale)
                    y, src = cam_yaw(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), depth, near=body_tgt)
                    fb.append({"pass": passes, "cam_yaw": y, "src": src, "err": None if y is None else body_tgt - y,
                               "odom_yaw": pose()[0], "live_samples": len(t2), "live_last": t2[-1][1] if t2 else None})
                    print(f"    live 패스 {passes}: 정지 명령 때 {fb[-1]['live_last']:+.2f}° → 멈춘 뒤 {src} {y:+.2f}° (오차 {fb[-1]['err']:+.2f})")
                print(f"    live: 샘플 {len(traj)} ({fb[0]['live_hz'] or 0:.1f} Hz) · 정지 명령 때 {fb[0]['live_last']:+.2f}° → "
                      f"멈춘 뒤 {src} {y:+.2f}° (목표 {body_tgt:+.2f}, 오차 {fb[0]['err']:+.2f})")
            elif args.feedback != "odom":
                for _ in range(args.feedback_iters):
                    color, depth = capture(cam, args.frames, scale)
                    y, src = cam_yaw(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), depth)
                    if y is None:
                        break
                    err = body_tgt - y
                    fb.append({"cam_yaw": y, "src": src, "err": err, "odom_yaw": pose()[0]})
                    if abs(err) < args.feedback_tol:
                        break
                    step = max(-10.0, min(10.0, err))
                    _drive_to(base, 1, math.radians(step), w, math.radians(5), abs(math.radians(step)) / w * 3 + 5)
                print("    피드백: " + " → ".join(f"{f['src']} {f['cam_yaw']:+.2f}° (오차 {f['err']:+.2f})" for f in fb))
            measure(i, f"목표 {tgt:+.0f}°")
            if fb and fb[-1]["cam_yaw"] is not None and abs(fb[-1]["err"]) < 1.5:
                anchors.append((fb[-1]["cam_yaw"], frames[-1]))  # 이 정지도 VO 기준 후보로
            stops[-1].update({"intended_deg": intended, "comp_deg": comp, "body_tgt_deg": body_tgt, "feedback": fb,
                              "live_traj": traj})
            last_sign, prev_tgt = sign, tgt
        if abs(fwd_total) > 0.01:
            print(f"  캐스터 정렬로 움직인 {fwd_total:+.3f} m 되돌린다")
            _drive_to(base, 0, -fwd_total, 0.08, 0.03, 15)
    finally:
        cleanup()
        cam.disconnect()

    # 정지 0 대비 회전량: depth vs 오도
    print("\n  정지   오도 Δyaw   depth Δyaw   depth/오도")
    rows = []
    for s in stops[1:]:
        do = s["odom_yaw_deg"] - stops[0]["odom_yaw_deg"]
        dd = s["depth_yaw_from0_deg"]
        ratio = dd / do if dd is not None and abs(do) > 3 else None
        rows.append({"stop": s["stop"], "odom_dyaw": do, "depth_dyaw": dd, "ratio": ratio})
        print(f"  {s['stop']:>3}   {do:+8.2f}°   {'—' if dd is None else f'{dd:+8.2f}°'}    {'—' if ratio is None else f'{ratio:.3f}'}")
    print("\n  이웃 정지끼리:  의도    전환보정   오도      VO      VO−의도")
    for a, b in zip(stops[:-1], stops[1:]):
        if "intended_deg" not in b:
            continue
        r = motion(*frames[a["stop"]], *frames[b["stop"]], K)
        b["vo_from_prev_deg"] = r["yaw_deg"]
        do = b["odom_yaw_deg"] - a["odom_yaw_deg"]
        vo = r["yaw_deg"]
        print(f"  {a['stop']}→{b['stop']}   {b['intended_deg']:+7.2f}  {b['comp_deg']:+6.1f}  {do:+7.2f}  "
              + ("    —" if vo is None else f"{vo:+7.2f}  {vo - b['intended_deg']:+6.2f}"))
    pos = [r["ratio"] for r in rows if r["ratio"] is not None and r["odom_dyaw"] > 0]
    neg = [r["ratio"] for r in rows if r["ratio"] is not None and r["odom_dyaw"] < 0]
    summary = {"turn_pos_depth_over_odom": float(np.median(pos)) if pos else None,
               "turn_neg_depth_over_odom": float(np.median(neg)) if neg else None}
    print(f"\n  depth/오도 중앙값: + 회전 {summary['turn_pos_depth_over_odom']} · − 회전 {summary['turn_neg_depth_over_odom']}")
    if args.dry_run:
        print("  (dry-run: 베이스는 가짜라 장면이 그대로 — VO yaw ≈ 0 이 정상. 흐름·노이즈 확인용)")
    (out / "report.json").write_text(json.dumps(
        {"args": vars(args), "K": K, "depth_scale": scale, "stops": stops, "rows": rows, "summary": summary},
        indent=2, ensure_ascii=False, default=float))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
