"""Does TrossenSlate.update_state() hold the GIL?  Read-only: no torque, no velocity command."""
import threading, time
from trossen_slate import TrossenSlate

base = TrossenSlate()
ok, msg = base.init_base()
print("init_base:", ok, msg.strip() if isinstance(msg, str) else msg, flush=True)
if not ok:
    raise SystemExit(1)

def count_for(seconds):
    n = 0; end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        n += 1
    return n / seconds

base_rate = count_for(2.0)
print(f"기준 (경쟁 없음): 파이썬 루프 {base_rate/1e6:.2f} M회/s", flush=True)

stop = threading.Event(); durs = []
def hammer():
    while not stop.is_set():
        t = time.perf_counter(); base.update_state(); durs.append(time.perf_counter() - t)
th = threading.Thread(target=hammer); th.start()
time.sleep(0.2)
rate = count_for(3.0)
stop.set(); th.join()
avg = sum(durs) / len(durs) * 1000
print(f"update_state() 연속 호출 중: 파이썬 루프 {rate/1e6:.2f} M회/s  ({rate/base_rate:.0%})", flush=True)
print(f"update_state() 1회 평균 {avg:.1f} ms, {len(durs)}회", flush=True)
print("판정:", "GIL을 쥔다 (다른 스레드의 파이썬이 멈춤)" if rate / base_rate < 0.5 else "GIL을 놓는다 (다른 스레드 영향 없음)")
