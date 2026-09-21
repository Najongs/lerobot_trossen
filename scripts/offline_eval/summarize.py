#!/usr/bin/env python
"""Collect every offline-eval output into markdown tables (docs 에 붙일 표).

usage: summarize.py <eval_dir>   (예: docs/run_logs/2026-09-22_offline_eval)
- chunks/*.npz  -> 열린 루프 정확도 (analyze_openloop) + 움직임 크기 (analyze_motion)
- sim/*.jsonl   -> 실행 방식 시뮬레이션 (simulate)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_openloop import analyze  # noqa: E402

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]


def motion(npz: Path) -> dict:
    d = np.load(npz)
    gt, pad, st = d["gt"], d["gt_pad"], d["state"]
    res = {}
    for key in [k for k in d.files if k.startswith("pred_")]:
        p = d[key]
        r = {}
        for h in (4, 9, 24):
            ok = ~pad[:, h]
            r[f"speed{h}"] = float(np.linalg.norm(p[ok, h][:, ARM] - st[ok][:, ARM], axis=1).mean()
                                   / np.linalg.norm(gt[ok, h][:, ARM] - st[ok][:, ARM], axis=1).mean())
        for h in (9, 24):
            ok = ~pad[:, h]
            g = gt[ok, h][:, ARM] - st[ok][:, ARM]
            q = p[ok, h][:, ARM] - st[ok][:, ARM]
            n = np.linalg.norm(g, axis=1)
            m = n > 0.05
            r[f"gain{h}"] = float(np.median((g[m] * q[m]).sum(1) / n[m] ** 2))
            r[f"cos{h}"] = float(np.median((g[m] * q[m]).sum(1) / (n[m] * np.linalg.norm(q[m], axis=1) + 1e-9)))
        res[key[5:]] = r
    return res


def main() -> None:
    root = Path(sys.argv[1])
    rows = []
    for f in sorted((root / "chunks").glob("*.npz")):
        if f.stem == "smoke":
            continue
        mo = motion(f)
        for r in analyze(f):
            r.update(mo.get(r["pred"], {}))
            rows.append(r)
    (root / "openloop_all.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cols = ["arm", "arm_h0", "arm_h20", "arm_h40", "grip", "base", "speed9", "gain9", "gain24", "cos24"]
    print("| 파일 | 설정 | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for r in rows:
        print(f"| {r['file']} | {r['pred']} | " + " | ".join(
            "" if r.get(c) is None else f"{r[c]:.3f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
