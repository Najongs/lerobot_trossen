"""CPU-only check of --ensemble.noise_scale / --ensemble.samples (no GPU, no robot)."""
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
    def __init__(self): self._queues = {ACTION: deque()}; self.seen = []
    def _prepare_batch(self, b): return b
    def parameters(self): yield torch.zeros(1)
    def _get_action_chunk(self, batch, noise):
        self.seen.append((None if noise is None else tuple(noise.shape),
                          None if noise is None else noise.std().item(),
                          tuple(batch["observation.state"].shape)))
        return (torch.ones(1, 50, 16) if noise is None else noise[:, :, :16].clone())

def run(**kw):
    R.install(0.01, 50, False, True, "off", False, **kw)
    f = Fake()
    a = SmolVLAPolicy.select_action(f, {"observation.state": torch.zeros(1, 16)})
    return f.seen[0], tuple(a.shape), a

seen, shape, a = run()
print("기본값           ->", seen, shape);             assert seen[0] is None and shape == (1, 16)
seen, shape, a = run(noise_scale=0.5)
print("noise_scale=0.5  ->", seen, shape);             assert seen[0] == (1, 50, 32) and 0.45 < seen[1] < 0.55
seen, shape, a = run(samples=4)
print("samples=4        ->", seen, shape);             assert seen[0] == (4, 50, 32) and seen[2] == (4, 16) and shape == (1, 16)
seen, shape, a = run(noise_scale=0.5, samples=4)
print("0.5 + samples=4  ->", seen, shape);             assert seen[0] == (4, 50, 32) and 0.45 < seen[1] < 0.55
seen, shape, a = run(noise_scale=0.0)
print("noise_scale=0    ->", seen, shape, "출력 전부 0:", bool((a == 0).all())); assert (a == 0).all()
print("ALL OK")
