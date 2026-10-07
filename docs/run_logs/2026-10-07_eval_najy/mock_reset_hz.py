"""mock 로봇·mock 정책으로 리셋 전용 체인을 돌려 러너 자체의 리셋 틱 주기(hertz)를 잰다 — 로봇 무접촉, I/O 없음."""
import sys, tempfile, json, statistics
from pathlib import Path
sys.path.insert(0, "packages/stage_runner/tests")
from test_chain_e2e import run_chain
with tempfile.TemporaryDirectory() as d:
    run = run_chain(Path(d), to_stage=11, extra_argv=("--chain.reset.only=true",))
    ends = run.of("stage_end")
    hz = [e.get("hertz") for e in ends if e.get("kind") == "reset" and e.get("hertz")]
    print("exit", run.exit_code, "| reset stages", len(hz), "| hertz 중앙", round(statistics.median(hz), 1), "범위", round(min(hz), 1), "~", round(max(hz), 1))
    print("frames/elapsed:", [(e.get("frames"), round(e.get("elapsed_s", 0), 2)) for e in ends][:4])
