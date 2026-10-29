#!/usr/bin/env python
"""캐스터(앞 보조 바퀴) 자세가 회전에 미치는 영향 — 같은 +15° 오도 회전을 네 가지 사전 자세에서 반복, 마커로 실제 회전 측정.

사전 자세: fwd = 앞 0.1 m 전진 뒤 (캐스터 뒤로 끌림·직선) · back = 뒤 0.1 m 후진 뒤 (캐스터 앞으로) ·
swungL = +10° 돌았다 −10° 돌아온 뒤 (캐스터 오른쪽 회전 자세) · swungR = −10° 돌았다 +10° 돌아온 뒤.
각 시행: 사전 자세 → 마커 측정 → +15° 오도 회전 → 마커 측정 → −15° 오도 복귀. 조건당 --reps 회.

  uv run python scripts/caster_test.py --pc pc2        # 🚨 베이스가 움직인다 (±15°, 앞뒤 0.1 m)
"""
import argparse, json, math, os, sys, time
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_approach import capture, open_cam
from depth_probe import SERIALS
from measure_base import _check_alive, _connect, _drive_to, _wrap
import marker_yaw as mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", default=os.environ.get("TROSSEN_PC", "pc1"), choices=list(SERIALS))
    ap.add_argument("--turn-deg", type=float, default=15.0)
    ap.add_argument("--speed-deg", type=float, default=15.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--conds", default="fwd,back,swungL,swungR")
    ap.add_argument("--out", default="outputs/caster_test")
    args = ap.parse_args()
    out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    cam, scale, K = open_cam(args.pc, 640, 480, 30)
    base, cleanup = _connect(False)
    w = math.radians(args.speed_deg)
    rows = []
    try:
        _check_alive(base, 1, probe=0.15)
        color, depth = capture(cam, 10, scale)
        up = mk.floor_up(depth, K)

        from depth_yaw import motion

        def yaw_now():
            """(마커 방위각 또는 None, 소스, (gray, depth)) — 마커가 안 보이면 호출 쪽이 VO 로 잰다."""
            color, depth = capture(cam, 10, scale)
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            m = mk.measure(gray, depth, up)
            for pref in (":flat", "pair"):
                for k in m:
                    if pref in k:
                        return m[k]["az_deg"], k, (gray, depth)
            return None, None, (gray, depth)

        def odom():
            base.update_state()
            return math.degrees(base.get_pose()[2])

        def turn(deg):
            return math.degrees(_drive_to(base, 1, math.radians(deg), w, math.radians(5), 10))

        prep = {
            "fwd": lambda: _drive_to(base, 0, 0.1, 0.08, 0.03, 8),
            "back": lambda: _drive_to(base, 0, -0.1, 0.08, 0.03, 8),
            "swungL": lambda: (turn(10), turn(-10)),
            "swungR": lambda: (turn(-10), turn(10)),
            # bf: 뒤 0.1 → 앞 0.1 — 마지막 동작이 전진(캐스터 상태 = fwd)이면서 마커는 원래 자리에서 보인다
            "bf": lambda: (_drive_to(base, 0, -0.1, 0.08, 0.03, 8), _drive_to(base, 0, 0.1, 0.08, 0.03, 8)),
            "fb": lambda: (_drive_to(base, 0, 0.1, 0.08, 0.03, 8), _drive_to(base, 0, -0.1, 0.08, 0.03, 8)),
        }
        conds = args.conds.split(",")
        for rep in range(args.reps):
            for c in conds:
                input(f"\n  [{c} #{rep + 1}] 사전 자세 → +{args.turn_deg}° → 복귀. 주변 비었으면 Enter: ")
                prep[c]()
                time.sleep(0.5)
                a0, src, f0 = yaw_now()
                o0 = odom()
                turn(args.turn_deg)
                a1, src1, f1 = yaw_now()
                o1 = odom()
                mo = motion(*f0, *f1, K)
                vo, vo_inl = mo["yaw_deg"], mo["inliers"]
                if a0 is not None and a1 is not None and src == src1:
                    real = -mk.wrap(a1 - a0, 360)
                else:
                    real, src = vo, "vo"
                tag = f"{c}_{rep}"
                cv2.imwrite(str(out / f"{tag}_pre.png"), f0[0]); np.save(out / f"{tag}_pre_depth.npy", f0[1])
                cv2.imwrite(str(out / f"{tag}_post.png"), f1[0]); np.save(out / f"{tag}_post_depth.npy", f1[1])
                do = _wrap(math.radians(o1 - o0))
                rows.append({"cond": c, "rep": rep, "odom_deg": math.degrees(do), "real_deg": real, "src": src, "vo_deg": vo})
                print(f"  {c:7s} 오도 {math.degrees(do):+6.2f}°  실제 {'—' if real is None else f'{real:+6.2f}°'} ({src})  "
                      f"VO {'—' if vo is None else f'{vo:+6.2f}'}(인라이어 {vo_inl})  지연 {'—' if real is None else f'{math.degrees(do) - real:+5.2f}'}")
                turn(-args.turn_deg)
                time.sleep(0.5)
    finally:
        cleanup()
        cam.disconnect()
    print("\n  조건별 (오도 − 실제) = 몸체가 덜 돈 양:")
    for c in conds:
        v = [r["odom_deg"] - r["real_deg"] for r in rows if r["cond"] == c and r["real_deg"] is not None]
        print(f"    {c:7s} {np.round(v, 2).tolist()}  평균 {np.mean(v):+.2f}" if v else f"    {c}: 측정 없음")
    (out / "report.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2, ensure_ascii=False))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
