"""CPU simulation. True trajectory: a(t) = t. Every predicted chunk is the truth plus its own
constant offset (sigma 3) -- i.e. chunks disagree, like independent flow-matching samples.
Jump = |a_t - a_(t-1) - 1|: zero on a perfectly smooth hand-over."""
import sys, time, types, random
from collections import deque
import torch
sys.path.insert(0, "/home/trossen-ai/NAJY/lerobot_trossen/scripts"); sys.argv = ["x"]
import record_ensemble as R
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
class Fake:
    training = False
    config = types.SimpleNamespace(chunk_size=50, max_action_dim=32)
    def __init__(self): self._queues = {ACTION: deque()}; self.n = 0; self.rng = random.Random(7)
    def _prepare_batch(self, b): return b
    def parameters(self): yield torch.zeros(1)
    def _get_action_chunk(self, batch, noise):
        self.n += 1; time.sleep(0.03)                         # 0.03 s at a 200 Hz sim loop = 6 steps of lag
        r = float(batch["t0"].item())
        return (r + torch.arange(50.0)).view(1, 50, 1).expand(1, 50, 16) + self.rng.gauss(0, 3)
def run(label, merge, commit, coeff=0.0):
    R.install(coeff, 1, True, True, "off", False, 1.0, 1, "thread", merge, commit)
    f = Fake(); a = []; blocks = 0
    for step in range(400):
        t0 = time.perf_counter()
        a.append(SmolVLAPolicy.select_action(f, {"t0": torch.tensor([step])})[0, 0].item())
        dt = time.perf_counter() - t0
        if step > 0 and dt > 0.004: blocks += 1
        time.sleep(max(0, 0.005 - dt))
    j = [abs(a[i] - a[i - 1] - 1) for i in range(1, len(a))]
    big = sum(1 for x in j if x > 1.0)
    print(f"{label:<34} 최대 점프 {max(j):5.2f} | RMS {(sum(x*x for x in j)/len(j))**0.5:5.3f} | 1 넘는 점프 {big:3d}회 | 추론 {f.n:3d}회 | 막힘 {blocks}")
run("commit=40, merge=latest (뚝 교체)", "latest", 40)
run("commit=40, merge=average (TA)", "average", 40)
run("commit=30, merge=average (TA)", "average", 30)
run("commit 없음, latest (뚝딱거린 구성)", "latest", 0)
run("commit 없음, average (연속 TA)", "average", 0)
