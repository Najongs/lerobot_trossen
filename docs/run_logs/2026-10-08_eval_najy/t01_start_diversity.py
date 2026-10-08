"""task01 시연 100개의 cam_high 첫 프레임이 서로 얼마나 다른지(시작 다양성) vs 실기 시작 프레임이 그 분포에서 얼마나 먼지. 파일만 읽음."""
import sys, pathlib, glob
import numpy as np
sys.argv = [sys.argv[0], "/dev/null"]
exec(pathlib.Path("docs/run_logs/2026-10-06_eval_najy/scene_rows.py").read_text().split("sheet = Image.new")[0])
C = "/home/trossen-ai/.cache/huggingface/lerobot/kiroaiseoul/"
def feat(root, e, f=0):
    im = frames_at(root, e, [f])[0].convert("L").resize((80, 60))
    return np.asarray(im, dtype=np.float32) / 255.0
T = C + "task01_move_to_tube_rack"
_, fps, ep, _ = load(T)
n = len(ep)
F = np.stack([feat(T, e) for e in range(n)])
D = np.sqrt(((F[:, None] - F[None]) ** 2).mean(-1).mean(-1)); np.fill_diagonal(D, np.inf)
nn = D.min(1)
print(f"task01 시연 {n}ep 첫 프레임끼리 최근접 거리(80x60 회색, RMS): 중앙 {np.median(nn):.3f} · p90 {np.percentile(nn,90):.3f} · 최대 {nn.max():.3f}")
# 같은 에피소드 안에서 시간이 흐르면 얼마나 변하나 (기준점): 0.5 s·1 s 뒤 프레임과의 거리
d05 = [np.sqrt(((feat(T, e, 15) - F[e]) ** 2).mean()) for e in range(0, n, 10)]
d10 = [np.sqrt(((feat(T, e, 30) - F[e]) ** 2).mean()) for e in range(0, n, 10)]
print(f"  참고: 같은 에피소드 0.5 s 뒤 프레임과의 거리 중앙 {np.median(d05):.3f}, 1 s 뒤 {np.median(d10):.3f}")
for name, repo in [("1040", "eval_1008_1040_chain_r4_t01_e30"), ("1046", "eval_1008_1046_chain_r4_t01_e30"), ("1047", "eval_1008_1047_chain_r4_t01_e30"), ("10/06 1347", "eval_najy_1006_1347_m1_t01_e30")]:
    try:
        g = feat(C + repo, 0, 0)
    except Exception as err:
        print(name, "읽기 실패", err); continue
    d = np.sqrt(((F - g) ** 2).mean(-1).mean(-1))
    print(f"실기 {name} 시작 프레임 → 시연 첫 프레임 최근접 {d.min():.3f} (시연끼리 중앙의 {d.min()/np.median(nn):.1f}배)")
