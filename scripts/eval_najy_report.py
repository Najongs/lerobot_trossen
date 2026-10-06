"""회차 한 장 보고서 — eval_najy_post.sh 가 만든 .summary/.motion/.latency/.base 와 로그·results.csv 를 표 하나로 모은다.

    uv run --no-sync python scripts/eval_najy_report.py <회차>      # markdown 을 stdout 으로

판정은 하지 않는다 — 수치만 모은다. 사람 판정은 ~/eval_logs/eval_najy_results.csv 의 해당 줄을 그대로 옮긴다.
없는 파일·못 읽은 값은 "-" 로 둔다 (실패를 숨기지 않으려고 어떤 파일이 없었는지 끝에 적는다).
베이스 구간(basevel 의 policy 구간)과 저장된 에피소드는 길이로 짝지운다 — ← 로 버린 시도는 구간만 있고 에피소드는 없다.
"""

import csv
import datetime as dt
import os
import re
import sys

LOGDIR = os.path.expanduser("~/eval_logs")
MODEL_NAME = {"m1": "M1 all11 s1000", "m2": "M2 all11 s2000", "m3": "M3 move4", "base": "단계 전문가(B*)"}


def read(path: str) -> str | None:
    return open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else None


def grab(pattern: str, text: str | None, group: int = 1, default: str = "-", flags=0):
    if text is None:
        return default
    m = re.search(pattern, text, flags)
    return m.group(group) if m else default


def main(run: str) -> int:
    missing = []
    log = read(f"{LOGDIR}/{run}.log")
    summary = read(f"{LOGDIR}/{run}.summary.txt")
    motion = read(f"{LOGDIR}/{run}.motion.txt")
    latency = read(f"{LOGDIR}/{run}.latency.txt")
    base = read(f"{LOGDIR}/{run}.base.txt")
    for name, text in (("log", log), ("summary", summary), ("motion", motion), ("latency", latency), ("base", base)):
        if text is None:
            missing.append(name)

    m = re.match(r"^(\d{2})(\d{2})_(\d{2})(\d{2})_([a-z0-9]+)_t(\d{2})_e(\d+)$", run)
    if not m:
        print(f"회차 이름 형식이 아니다: {run}", file=sys.stderr)
        return 2
    mm, dd, hh, mi, tag, stage, ex = m.groups()
    model = MODEL_NAME.get(tag, tag)

    # --- 로그
    onehot = grab(r"LEROBOT_TASK_ONEHOT=(\S+) installed", log, default="없음(원핫 모델 아님)" if tag == "base" else "**없음 — 무효**")
    active = "✓" if log and re.search(r"stage \d+/\d+ active", log) else ("-" if tag == "base" else "**없음**")
    rates = re.findall(r"Control loop rate .*phase=policy.*?mean=([\d.]+) Hz, min=([\d.]+) Hz", log or "")
    rate_s = f"mean {rates[-1][0]} Hz (min {rates[-1][1]})" if rates else "-"
    # policy 구간만. phase 끝 줄(record_loop 한 번 = 에피소드 하나의 전체 합)이 완전한 값; 없으면(중단·옛 포맷) 창 합으로 —
    # eval_najy_post.sh 의 요약과 같은 규칙.
    phase_totals = [int(x) for x in re.findall(r"clamps this policy phase: (\d+) arm-ticks", log or "")]
    win_total = sum(int(x) for x in re.findall(r"phase=policy.*?clamped=(\d+)", log or ""))
    clamp_warn = len(re.findall(r"had to be clamped", log or ""))
    if phase_totals:
        clamp_s = f"{sum(phase_totals)} arm-tick (에피소드별 {'/'.join(map(str, phase_totals))})"
    else:
        clamp_s = f"{win_total} arm-tick (창 합 — phase 끝 줄 없음)"
    if clamp_warn:
        clamp_s += f" (+필터 전 경고 {clamp_warn}줄)"
    fired = len(re.findall(r"\bFIRED\b", log or ""))
    rc = grab(r"종료 코드 (\S+)", summary)
    max_rel = grab(r"팔 한 틱 상한 (\S+)", summary)
    estop = "⚠️ 비상정지/베이스 경고 있음" if log and re.search(r"emergency|E-STOP|estop (triggered|active)", log, re.I) else ""

    # 에피소드 시작 자세: POSE-START(첫 정책 틱) 우선, 없으면 리셋 마지막 POSE 줄. 둘 다 없는 ep0 는 "-"
    # (ep0 앞엔 리셋이 없다 — POSE-START 는 10/06 오후부터 찍힌다).
    start_pose = {}
    if log:
        pose, cur = None, None
        for line in log.splitlines():
            if " POSE task" in line:
                pose = line
            mm_ep = re.search(r"Recording episode (\d+)", line)
            if mm_ep:
                cur = int(mm_ep.group(1))
                if pose:
                    start_pose[cur] = (grab(r"목표까지 ([\d.]+) rad", pose), grab(r"가장 큰 차 ([^|]+?) \|", pose).strip() + " (리셋 끝)")
                pose = None
            if " POSE-START " in line and cur is not None:
                start_pose[cur] = (grab(r"목표까지 ([\d.]+) rad", line), grab(r"가장 큰 차 ([^|]+?) \|", line).strip())

    # --- motion
    train_line = grab(r"^(학습 .*)$", motion, flags=re.M)
    allow = grab(r"시작점끼리 최근접 거리 중앙값 ([\d.]+)", motion)
    train_move = grab(r"프레임당 이동 중앙값 ([\d.]+)", motion)
    eps = {}
    for em in re.finditer(
        r"ep(\d+): ([\d.]+)s, (\d+) fps \| 프레임당 이동 ([\d.]+) rad, 방향 바뀜 ([\d.]+)/s \| 시작점 최근접 ([\d.]+) "
        r"\| 청크 경계 점프 ([\d.]+) / 그 외 ([\d.]+) \(([\d.]+)x",
        motion or "",
    ):
        ep = int(em.group(1))
        eps[ep] = dict(dur=float(em.group(2)), move=em.group(4), turn=em.group(5), start=float(em.group(6)),
                       jump=em.group(7), rest=em.group(8), ratio=em.group(9))

    # --- base
    segs = []
    for sm in re.finditer(
        r"seg(\d+): +([\d.]+)s (\d+)행 \| 회전 명령 +([+-][\d.]+)° 실측 +([+-][\d.]+)° \| 전진 명령 +([+-][\d.]+) m 실측 +([+-][\d.]+) m",
        base or "",
    ):
        segs.append(dict(dur=float(sm.group(2)), rot=f"{sm.group(4)} / {sm.group(5)}", fwd=f"{sm.group(6)} / {sm.group(7)}"))
    train_rot = grab(r"회전 중앙 ([+-]\d+)°", base)
    train_fwd = grab(r"전진 중앙 ([+-][\d.]+) m", base)
    train_dur = grab(r"길이 중앙 ([\d.]+)s", base)
    train_21 = grab(r"실제 녹화가 21 Hz 였다면 \(×[\d.]+\): (.*?) —", base)
    # 에피소드 ↔ 구간: 둘 다 시간 순이므로 순서를 지키며 앞에서부터 짝짓는다. 에피소드 길이는 frames/fps(공칭)이고
    # 구간 길이는 실측이라 루프가 느리면 120초에서 1초 넘게 어긋난다 → 허용 max(1.5초, 3%). 건너뛴 구간 = ← 로 버린 시도.
    used = set()
    ep_seg = {}
    j = 0
    for ep, e in sorted(eps.items()):
        tol = max(1.5, 0.03 * e["dur"])
        while j < len(segs) and abs(segs[j]["dur"] - e["dur"]) >= tol:
            j += 1
        if j < len(segs):
            used.add(j)
            ep_seg[ep] = segs[j]
            j += 1
    orphan_segs = [s for i, s in enumerate(segs) if i not in used]

    # --- latency
    arm_lag = grab(r"팔 \(eval \d+ep\): 지연 (\d+) 틱", latency)
    arm_train = grab(r"학습 .*?: 지연 (\d+) 틱", latency)
    spike = grab(r"추론 스파이크 .*?중앙값 (\d+) ms", latency)
    base_rot_lag = grab(r"회전: 지연 (\d+ 틱 ≈ \d+ ms)", latency)
    base_fwd_lag = grab(r"전진: (지연 \d+ 틱 ≈ \d+ ms|명령이 거의 0[^|\n]*)", latency)
    residual = grab(r"정지 뒤 잔여 회전: (.*)", latency)

    # --- 사람 판정 (results.csv)
    verdict = {}
    csv_path = f"{LOGDIR}/eval_najy_results.csv"
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 4 and row[0] == run:
                    verdict[row[1]] = (row[2], row[3])
    else:
        missing.append("results.csv")

    # --- 출력
    out = []
    out.append(f"# {run} — {model} · task{stage} · exec {ex}")
    today = dt.date.today()
    year = today.year - (1 if f"{mm}{dd}" > today.strftime("%m%d") else 0)  # 회차 이름엔 연도가 없다
    out.append(f"{year}-{mm}-{dd} {hh}:{mi} (Trossen PC1) · 원핫 {onehot} {active} · 팔 한 틱 상한 {max_rel} · 종료 코드 {rc} {estop}")
    out.append("")
    out.append(f"- 루프: policy {rate_s} · 상한에 잘린 틱 {clamp_s} · 페이싱 FIRED {fired} · 추론 스파이크 중앙 {spike} ms")
    out.append(f"- 지연: 팔 {arm_lag}틱 (학습 {arm_train}틱) · 베이스 회전 {base_rot_lag} / 전진 {base_fwd_lag}"
               + (f" · 정지 뒤 잔여 회전 {residual}" if residual != "-" else "")
               + ("" if int(stage) in (1, 3, 6, 10) else " — 조작 단계라 베이스 계단이 적어 베이스 지연은 신뢰 낮음"))
    out.append(f"- 저장된 에피소드 {len(eps)} · 베이스 정책 구간 {len(segs)}" + (f" (짝 없는 구간 {len(orphan_segs)}: 시작 직후 버린 시도일 수 있음)" if orphan_segs else ""))
    out.append("")
    out.append("| ep | 길이 | 시작 직전 POSE (중앙값까지 / 가장 큰 차) | 시작점 최근접 (허용) | 팔 이동/프레임 (학습) | 방향 바뀜/s | 경계 점프 / 그 외 (배) | 베이스 회전 명령/실측 ° | 전진 명령/실측 m | 사람 판정 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    all_eps = sorted(set(eps) | {int(k) for k in verdict if k.isdigit()})
    for ep in all_eps:
        e = eps.get(ep)
        sp = start_pose.get(ep)
        sp_s = f"{sp[0]} rad / {sp[1]}" if sp else "-"
        if e:
            flag = "" if allow == "-" or e["start"] <= float(allow) else " ⚠️"
            start_s = f"{e['start']:.2f} ({allow}){flag}"
            move_s = f"{e['move']} ({train_move})"
            jump_s = f"{e['jump']} / {e['rest']} ({e['ratio']}×)"
            dur_s = f"{e['dur']:.1f}s"
            turn_s = e["turn"]
        else:
            start_s = move_s = jump_s = dur_s = turn_s = "-"
        s = ep_seg.get(ep)
        rot_s = f"{s['rot']}" if s else "-"
        fwd_s = f"{s['fwd']}" if s else "-"
        v = verdict.get(str(ep))
        v_s = (("✅ " if v[0] == "1" else "❌ " if v[0] == "0" else f"{v[0]} ") + v[1]) if v else "(미기록 — results.csv)"
        out.append(f"| {ep} | {dur_s} | {sp_s} | {start_s} | {move_s} | {turn_s} | {jump_s} | {rot_s} | {fwd_s} | {v_s} |")
    for s in orphan_segs:
        out.append(f"| (구간만) | {s['dur']:.1f}s | | | | | | {s['rot']} | {s['fwd']} | 저장 안 됨 |")
    out.append("")
    if train_line != "-":
        out.append(f"학습 기준 — {train_line}")
    if train_rot != "-":
        out.append(f"학습 베이스 — 회전 {train_rot}° · 전진 {train_fwd} m · 길이 {train_dur}s (표기 fps 기준; 21 Hz 였다면 {train_21})")
    if missing:
        out.append("")
        out.append(f"없던 입력: {', '.join(missing)}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
