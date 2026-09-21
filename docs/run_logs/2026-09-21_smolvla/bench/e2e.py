"""End-to-end check of the async worker under the robot's real GIL pattern.
The 'control loop' calls base.update_state() twice per step (read-only; ~42-50 ms with the
GIL held, like base_read + base_write on the robot) and then policy.select_action()."""
import sys, time
sys.path.insert(0, "/home/trossen-ai/NAJY/lerobot_trossen/scripts")

def main(KIND, STEPS):
    import torch
    import record_ensemble as R
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    from trossen_slate import TrossenSlate

    P = "/home/trossen-ai/daehee/models/smolvla_190k_30k_aug_80k_kirogist/pretrained_model"
    cfg = PreTrainedConfig.from_pretrained(P); cfg.pretrained_path = P; cfg.num_steps = 10
    stats = R.install(0.01, 1, True, True, "bf16", True, 0.5, 4, KIND)
    dev = torch.device("cuda:0")
    policy = SmolVLAPolicy.from_pretrained(P, config=cfg).to(dev).eval()

    base = TrossenSlate(); ok, msg = base.init_base(); assert ok, msg
    L = 20
    def obs():
        b = {OBS_STATE: torch.zeros(1, 16, device=dev),
             OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, L), device=dev),
             OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, L, dtype=torch.bool, device=dev)}
        for k in cfg.input_features:
            if "images" in k: b[k] = torch.rand(1, 3, 480, 640, device=dev)
        return b

    policy.reset()
    print('LOOP_START', time.time(), flush=True)
    dts, sel = [], []
    with torch.inference_mode():
        for i in range(STEPS):
            t0 = time.perf_counter()
            base.update_state(); base.update_state()          # GIL held, like base_read + base_write
            t1 = time.perf_counter()
            a = policy.select_action(obs()); a.cpu()
            t2 = time.perf_counter()
            if i > 0: dts.append(t2 - t0); sel.append(t2 - t1)
    print('LOOP_END', time.time(), flush=True)
    print(f"[{KIND}] 제어루프 평균 {len(dts)/sum(dts):.1f} Hz | select_action 평균 {sum(sel)/len(sel)*1000:.1f} ms, 최악 {max(sel)*1000:.0f} ms (첫 스텝 제외)")
    print(f"[{KIND}] {stats.report()}")

if __name__ == "__main__":
    kind, steps = sys.argv[1], int(sys.argv[2])
    sys.argv = ["x"]
    main(kind, steps)
