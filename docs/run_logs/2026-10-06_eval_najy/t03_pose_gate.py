"""task03: 시연에서 회전이 시작되는 순간의 팔 자세와 실기 에피소드의 팔 자세 거리 (파일만 읽음)."""
import glob, numpy as np, pandas as pd
C = "/home/trossen-ai/.cache/huggingface/lerobot/kiroaiseoul/"
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
NM = ["L0", "L1", "L2", "L3", "L4", "L5", "R0", "R1", "R2", "R3", "R4", "R5"]
def load(root):
    d = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "action", "observation.state"]) for p in sorted(glob.glob(root + "/data/*/*.parquet"))])
    return d.sort_values(["episode_index", "frame_index"])
tr = load(C + "task03_turn_to_face_beaker")
S0, SON, ALL, ON, GR = [], [], [], [], []
for e, g in tr.groupby("episode_index"):
    a = np.stack(g["action"].to_numpy()); s = np.stack(g["observation.state"].to_numpy())
    on = int(np.argmax(np.abs(a[:, 15]) > 0.05))
    S0.append(s[0, ARM]); SON.append(s[on, ARM]); ON.append(on); ALL.append(s[: on + 1, ARM]); GR.append(s[0, [6, 13]])
S0, SON = np.array(S0), np.array(SON); PRE = np.concatenate(ALL)
print(f"학습 task03 {len(S0)}ep | 회전 시작 프레임 중앙 {np.median(ON):.0f} (p25 {np.percentile(ON,25):.0f} / p75 {np.percentile(ON,75):.0f} / 최대 {max(ON)})")
print("  시작 자세 중앙(도):     " + " ".join(f"{n}{v:+.0f}" for n, v in zip(NM, np.degrees(np.median(S0, 0)))))
print("  회전 시작 자세 중앙(도): " + " ".join(f"{n}{v:+.0f}" for n, v in zip(NM, np.degrees(np.median(SON, 0)))))
print(f"  시작→회전시작 팔 이동 L2 중앙 {np.median(np.linalg.norm(SON - S0, axis=1)):.2f} rad | 회전시작 자세끼리 최근접 중앙 {np.median([np.sort(np.linalg.norm(SON - x, axis=1))[1] for x in SON]):.2f}")
print(f"  그리퍼 시작 중앙 {np.median(np.array(GR),0).round(4)}")
ev = load(C + "eval_najy_1006_1350_m1_t03_e30")
med_on = np.median(SON, 0)
for e, g in ev.groupby("episode_index"):
    a = np.stack(g["action"].to_numpy()); s = np.stack(g["observation.state"].to_numpy()); q = s[:, ARM]
    on = int(np.argmax(np.abs(a[:, 15]) > 0.05)) if (np.abs(a[:, 15]) > 0.05).any() else -1
    print(f"-- 1350 ep{e}: {len(s)}프레임, 회전 명령 시작 프레임 {on} ({on/21:.1f}s)" if on >= 0 else f"-- 1350 ep{e}: {len(s)}프레임, 회전 명령 없음 (|θ| 최대 {np.abs(a[:,15]).max():.3f})")
    print(f"   그리퍼 시작 {s[0,[6,13]].round(4)} 끝 {s[-1,[6,13]].round(4)}")
    ts = sorted(set([0, 105, 210, 315] + ([on] if on >= 0 else []) + [420, 630, 1050, 1470, len(s) - 1]))
    for t in [t for t in ts if t < len(s)]:
        d0 = np.linalg.norm(S0 - q[t], axis=1).min(); don = np.linalg.norm(SON - q[t], axis=1).min(); dpre = np.linalg.norm(PRE - q[t], axis=1).min()
        diff = np.degrees(q[t] - med_on); k = np.argsort(-np.abs(diff))[:3]
        print(f"   t={t/21:5.1f}s 최근접: 시작 {d0:.2f} · 회전시작 {don:.2f} · 회전 전 구간 전체 {dpre:.2f} | 회전시작 중앙 대비 큰 차: " + " ".join(f"{NM[i]}{diff[i]:+.0f}°" for i in k) + f" | θcmd {a[t,15]:+.2f}")
