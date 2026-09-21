#!/usr/bin/env python
"""Does the policy move as far as the demonstrator? ("방향은 맞는데 끝까지 못 간다" 의 정량화)

시연이 호라이즌 h 동안 팔을 충분히 움직인 프레임(|Δ| > thr rad)에서
- cos: 예측 변위와 시연 변위의 방향 코사인 (팔 12축 벡터)
- gain: 예측 변위를 시연 변위 방향에 사영한 길이 / 시연 변위 길이. 1 이면 같은 만큼, <1 이면 덜 간다
- 그리퍼: 시연이 닫는(액션이 0.005 m 이상 줄어드는) 구간에서 예측이 닫는 양의 비율
변위는 모두 현재 state 기준.
"""
import json, sys
import numpy as np
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]
out = []
for f in sys.argv[1:]:
    d = np.load(f)
    gt, pad, st = d["gt"], d["gt_pad"], d["state"]
    for key in [k for k in d.files if k.startswith("pred_")]:
        p = d[key]
        row = {"file": f.split("/")[-1][:-4], "pred": key[5:]}
        for h in (9, 24, 49):
            ok = ~pad[:, h]
            g = gt[ok, h][:, ARM] - st[ok][:, ARM]
            q = p[ok, h][:, ARM] - st[ok][:, ARM]
            n = np.linalg.norm(g, axis=1)
            m = n > 0.05
            g, q, n = g[m], q[m], n[m]
            cos = (g * q).sum(1) / (n * np.linalg.norm(q, axis=1) + 1e-9)
            gain = (g * q).sum(1) / n ** 2
            row[f"cos{h}"] = float(np.median(cos)); row[f"gain{h}"] = float(np.median(gain)); row[f"n{h}"] = int(m.sum())
        # 속도 비율: 모든 프레임에서 |예측 변위| 평균 / |시연 변위| 평균 (방향 무관, 얼마나 빨리 가려 하나)
        for h in (4, 9, 24):
            ok = ~pad[:, h]
            row[f"speed{h}"] = float(np.linalg.norm(p[ok, h][:, ARM] - st[ok][:, ARM], axis=1).mean()
                                     / np.linalg.norm(gt[ok, h][:, ARM] - st[ok][:, ARM], axis=1).mean())
        # gripper closing
        ok = ~pad[:, 24]
        gg = gt[ok, 24][:, GRIP] - st[ok][:, GRIP]
        pg = p[ok, 24][:, GRIP] - st[ok][:, GRIP]
        close = gg < -0.005
        row["grip_close_gain"] = float(np.median(pg[close] / gg[close])) if close.any() else float("nan")
        row["grip_close_n"] = int(close.sum())
        out.append(row)
COLS = ["speed4", "speed9", "speed24", "cos9", "gain9", "cos24", "gain24", "gain49", "grip_close_gain"]
print(f"{'file':<40}{'pred':<9}" + "".join(f"{c[:8]:>9}" for c in COLS))
for r in out:
    print(f"{r['file']:<40}{r['pred']:<9}" + "".join(f"{r[c]:>9.3f}" for c in COLS))
if len(sys.argv) > 1:
    import os
    with open(os.environ.get("MOTION_OUT", "/dev/null"), "a") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
