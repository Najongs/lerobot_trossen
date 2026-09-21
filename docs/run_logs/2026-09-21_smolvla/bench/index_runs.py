import glob, os, re, sys, time
rows = []
for L in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "*.log")), key=os.path.getmtime):
    txt = open(L, errors="replace").read(); lines = txt.splitlines()
    g = lambda pat, d="": (re.search(pat, txt).group(1) if re.search(pat, txt) else d)
    start = g(r"(\d\d:\d\d:\d\d) \S+ (?:샘플링 조정|temporal ensembling|commit=)", g(r"(\d\d:\d\d:\d\d)"))
    act = next((l for l in lines if "temporal ensembling 활성화" in l), "")
    sync = bool(re.search(r"every=\d+, 동기\)", act))
    mode = "동기" if sync else ("비동기/" + (re.search(r"async/(\w+)", act).group(1) if re.search(r"async/(\w+)", act) else "thread"))
    coeff = g(r"활성화 \(coeff=([-\d.e]+)"); every = g(r"활성화 \(coeff=[-\d.e]+, every=(\d+)")
    amp = "로그에 없음" if sync else g(r"amp=(\w+)\)")
    commit = g(r"commit=(\d+):", "-")
    merge = "latest" if ("청크 병합: latest" in txt or "통째로 갈아탄다" in txt) else "average"
    ns = g(r"초기 노이즈 x([\d.]+)", "1"); K = g(r"노이즈 샘플 (\d+)개", "1")
    steps = g(r"'num_steps': (\d+)"); m = re.search(r"'single_task': ((?:'[^']*'\s*)+)", txt); task = "".join(re.findall(r"'([^']*)'", m.group(1))).strip() if m else ""
    inf = re.findall(r"추론 (\d+)회 · 평균 ([\d.]+) ms \(최악 ([\d.]+) ms\)", txt)
    lag = re.findall(r"비행 중 지연 평균 ([\d.]+) 스텝 \(최악 (\d+)\)", txt)
    hz = [float(x) for x in re.findall(r"phase=policy[^\n]*?mean=([\d.]+)", txt)]
    phase = None; cp = 0
    for l in lines:
        if "Recording episode" in l: phase = "policy"
        elif "Reset the environment" in l: phase = "reset"
        elif "had to be clamped" in l and phase == "policy": cp += 1
    basefail = len(re.findall(r"base transaction failed|Failed to refresh Mobile AI base", txt))
    rerec = txt.count("rerecord the last episode"); 
    note = []
    if basefail: note.append(f"base 통신 실패 {basefail}회")
    if rerec: note.append(f"왼쪽 화살표 {rerec}회")
    if not inf: note.append("추론 요약 줄 없음(tee가 끊김 또는 시작 전 종료)")
    rows.append(dict(f=os.path.basename(L), d=g(r"2026-(\d\d-\d\d) \d\d:\d\d:\d\d"), t=start, task=task, steps=steps, mode=mode,
                     coeff=coeff, every=every, commit=commit, merge=merge, ns=ns, K=K, amp=amp,
                     n=inf[-1][0] if inf else "", ms=inf[-1][1] if inf else "", worst=inf[-1][2] if inf else "",
                     lag=f"{lag[-1][0]} / {lag[-1][1]}" if lag else "",
                     hz=f"{sum(hz)/len(hz):.1f}" if hz else "", sec=f"{len(hz)*30/(sum(hz)/len(hz)):.0f}" if hz else "",
                     cp=str(cp), note=", ".join(note)))
tasks = {}
for r in rows: tasks.setdefault(r["task"], f"T{len(tasks)+1}"); r["tk"] = tasks[r["task"]]
out = ["| # | 날짜 시각 | 과제 | 모드 | num_steps | coeff | every | commit | merge | 노이즈 | K | amp | 추론 횟수 | 평균 ms (최악) | 지연 평균 / 최악 | 정책 구간 Hz | 정책 구간 초 | 정책 구간 클램프 | 비고 | 관찰(직접 기입) |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for i, r in enumerate(rows, 1):
    ms = f"{r['ms']} ({r['worst']})" if r["ms"] else ""
    out.append(f"| R{i} | {r['d']} {r['t']} | {r['tk']} | {r['mode']} | {r['steps']} | {r['coeff']} | {r['every']} | {r['commit']} | {r['merge'] if r['commit']!='-' or 'latest'==r['merge'] else 'average'} | {r['ns']} | {r['K']} | {r['amp']} | {r['n']} | {ms} | {r['lag']} | {r['hz']} | {r['sec']} | {r['cp']} | {r['note']} | |")
print("\n".join(out)); print()
for t, k in tasks.items(): print(f"- {k}: \"{t}\"")
print(); print("FILES"); 
for i, r in enumerate(rows, 1): print(f"R{i}\t{r['f']}")
