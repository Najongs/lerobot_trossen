#!/usr/bin/env python
"""RMSE 를 최소로 만드는 변위 배율 α: action' = state + α (pred - state).
불편(unbiased) 예측이면 α* ~ 1 이고, α* > 1 이면 예측 변위가 체계적으로 작다(위축)."""
import sys, numpy as np
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
BANDS = [(0, 10), (10, 25), (25, 50)]
for f in sys.argv[1:]:
    d = np.load(f)
    gt, pad, st = d["gt"], d["gt_pad"], d["state"]
    for key in [k for k in d.files if k.startswith("pred_")]:
        p = d[key]
        out = []
        for a, b in BANDS:
            ok = ~pad[:, a:b]
            q = (p[:, a:b][..., ARM] - st[:, None, ARM])[ok]
            g = (gt[:, a:b][..., ARM] - st[:, None, ARM])[ok]
            alpha = float((q * g).sum() / (q * q).sum())
            r1 = np.sqrt(((q - g) ** 2).mean()); ra = np.sqrt(((alpha * q - g) ** 2).mean())
            out.append(f"h{a}-{b - 1}: a*={alpha:.2f} rmse {r1:.4f}->{ra:.4f}")
        print(f"{f.split('/')[-1][:-4]:<34}{key[5:]:<8}" + " | ".join(out))
