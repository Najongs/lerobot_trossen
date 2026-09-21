import sys, types
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
        self.n += 1; return torch.full((1, 50, 16), float(self.n))   # chunk k is all k's
def run(merge):
    R.install(0.0, 2, False, True, "off", False, 1.0, 1, "thread", merge)  # sync, re-query every 2 steps
    f = Fake(); out = [SmolVLAPolicy.select_action(f, {"observation.state": torch.zeros(1, 16)})[0, 0].item() for _ in range(6)]
    return out
a, l = run("average"), run("latest")
print("average:", a); print("latest :", l)
assert l == [1, 1, 2, 2, 3, 3], l            # pure replacement
assert a[2] == 1.5 and a[4] == 2.0, a        # equal-weight mean of overlapping chunks
print("ALL OK")
