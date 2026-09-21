#!/usr/bin/env python
"""Open-loop chunk accuracy from ``predict_chunks.py`` outputs.

지표 (모두 실제 단위, 시연 액션 대비):
- 팔 RMSE (rad) 전체 / 호라이즌 구간별 (0-9, 10-19, 20-29, 30-39, 40-49)
- 그리퍼 RMSE (m), 베이스 RMSE (m/s, rad/s)
- 기준선 "현재 자세 유지": 청크 전체를 현재 state 로 채운 것. 정책이 이보다 얼마나 나은가
- 진행 구간별(에피소드 길이의 5분위) 팔 RMSE (호라이즌 0-9)
- 파지 구간 그리퍼 편향: 시연 그리퍼 액션이 state 보다 0.001 m 이상 작은(쥐고 조이는) 프레임에서
  예측 - 시연 의 평균. 양수면 정책이 덜 쥔다
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]
BASE = [14, 15]
BANDS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]


def rmse(e2: np.ndarray, mask: np.ndarray, dims) -> float:
    sel = e2[..., dims][mask]
    return float(np.sqrt(sel.mean())) if sel.size else float("nan")


def analyze(npz: Path) -> list[dict]:
    d = np.load(npz)
    gt, pad, st, ep, fr = d["gt"], d["gt_pad"], d["state"], d["episode"], d["frame"]
    ok = ~pad
    # 에피소드 길이 -> 진행도
    length = {e: fr[ep == e].max() + 1 for e in np.unique(ep)}
    prog = np.array([f / length[e] for e, f in zip(ep, fr)])
    hold = np.repeat(st[:, None], gt.shape[1], axis=1)
    # 파지: 시연이 그리퍼를 state 보다 더 닫으라고 명령하는 (첫 스텝) 프레임
    squeeze = (gt[:, 0, GRIP] < st[:, GRIP] - 0.001)
    rows = []
    for key in ["hold"] + [k for k in d.files if k.startswith("pred_")]:
        p = hold if key == "hold" else d[key]
        e2 = (p - gt) ** 2
        r = {"file": npz.stem, "pred": key.removeprefix("pred_"), "n": int(len(gt))}
        r["arm"] = rmse(e2, ok, ARM)
        r["grip"] = rmse(e2, ok, GRIP)
        r["base"] = rmse(e2, ok, BASE)
        for a, b in BANDS:
            m = ok.copy()
            m[:, :a] = False
            m[:, b:] = False
            r[f"arm_h{a}"] = rmse(e2, m, ARM)
        for q in range(5):
            sel = (prog >= q / 5) & (prog < (q + 1) / 5)
            m = ok.copy()
            m[~sel] = False
            m[:, 10:] = False
            r[f"arm_p{q}"] = rmse(e2, m, ARM)
        bias = []
        for gi, g in enumerate(GRIP):
            s = squeeze[:, gi]
            if s.any():
                bias.append(float((p[s, 0, g] - gt[s, 0, g]).mean()))
        r["grip_squeeze_bias"] = float(np.mean(bias)) if bias else float("nan")
        r["squeeze_frames"] = int(squeeze.any(1).sum())
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for f in args.npz:
        rows += analyze(Path(f))
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cols = ["arm", "arm_h0", "arm_h20", "arm_h40", "grip", "base", "arm_p0", "arm_p1", "arm_p2", "arm_p3", "arm_p4", "grip_squeeze_bias"]
    print(f"{'file':<48}{'pred':<7}" + "".join(f"{c:>10}" for c in cols))
    for r in rows:
        print(f"{r['file']:<48}{r['pred']:<7}" + "".join(f"{r[c]:>10.4f}" for c in cols))


if __name__ == "__main__":
    main()
