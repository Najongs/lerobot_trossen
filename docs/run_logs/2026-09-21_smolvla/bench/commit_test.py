"""Simulate async + every=N + merge=latest on CPU: fake policy whose inference takes 0.3 s,
control loop at 20 Hz. Which chunk does each executed step come from, and does the loop ever block?"""
import sys, time, types
from collections import deque
import torch
sys.path.insert(0, "/home/trossen-ai/NAJY/lerobot_trossen/scripts"); sys.argv = ["x"]
import record_ensemble as R
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
class Fake:
    training = False
    config = types.SimpleNamespace(chunk_size=50, max_action_dim=32)
    def __init__(self): self._queues = {ACTION: deque()}; self.n = 0
    def _prepare_batch(self, b): return b
    def parameters(self): yield torch.zeros(1)
    def _get_action_chunk(self, batch, noise):
        self.n += 1; time.sleep(0.3)                     # 0.3 s inference = 6 steps at 20 Hz
        k = torch.arange(50).view(1, 50, 1).expand(1, 50, 16)
        return self.n * 1000.0 + k                        # value = chunk_id*1000 + index within chunk
for commit in (40, 50):
    every = 1
    R.install(0.0, 1, True, True, "off", False, 1.0, 1, "thread", "average", commit)
    f = Fake(); rows = []; blocks = []
    for step in range(170):
        t0 = time.perf_counter()
        v = SmolVLAPolicy.select_action(f, {"observation.state": torch.zeros(1, 16)})[0, 0].item()
        dt = time.perf_counter() - t0
        if step > 0 and dt > 0.04: blocks.append((step, round(dt * 1000)))
        rows.append((int(v // 1000), int(v % 1000)))
        time.sleep(max(0, 0.05 - dt))
    runs = []; 
    for c, i in rows:
        if runs and runs[-1][0] == c: runs[-1][2] = i; runs[-1][3] += 1
        else: runs.append([c, i, i, 1])
    print(f"commit={commit}: " + " | ".join(f"청크{c}: 인덱스 {a}~{b} ({n}스텝)" for c, a, b, n in runs))
    print(f"          추론 {f.n}회 |  제어 루프가 막힌 스텝(첫 스텝 제외): {blocks if blocks else '없음'}")
