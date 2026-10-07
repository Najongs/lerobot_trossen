#!/usr/bin/env python3
"""체인 한 회차의 `events.jsonl` 을 단계별 표 한 장으로 만든다.

    uv run python scripts/eval_chain_report.py outputs/stage_runner/<회차> \\
        [--basevel ~/eval_logs/<회차>.basevel.csv] [--out <파일>]

왜 `eval_najy_post.sh` / `eval_najy_report.py` 를 고치지 않나: 그 둘은
**회차 = 단계 하나**를 가정한다(파일명·요약·판정표 전부). 체인은 한 회차에
22단계가 한 에피소드로 들어 있어서, 같은 스크립트에 넣으면 두 전제가 한 파일에
섞인다. 이 스크립트는 체인 전용이고 그 둘을 수정 0 으로 둔다.

표준 라이브러리만 쓴다. `aggregate.py` 와 같은 이유다 -- 몇 달 뒤 로봇 환경이
옮겨간 뒤에도 남아 있는 `events.jsonl` 을 다시 읽을 수 있어야 한다. basevel CSV
도 `csv` 모듈로 읽는다(pandas 없이).

베이스 적분은 `scripts/eval_base_stats.py:45-52` 의 수식을 그대로 쓴다:
행마다 Δt 를 다음 행까지의 시간으로 잡고(마지막은 중앙값) 속도에 곱해 더한다.
단계별 범위는 `stage_start`/`stage_end` 의 **`t_mono`** 로 자른다 -- 그 값은
`perf_counter` 이고 basevel.csv 의 `t_mono` 열과 같은 시계다(base_vel_log.py:69).
`wall_clock_iso` 로는 못 한다: 밀리초 해상도의 지역시각이고 원점이 다르다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

# 터미널에 바로 쓰는 기본값. --out 을 주면 파일로도 쓴다.
STAGE_KIND_POLICY = "policy"
STAGE_KIND_RESET = "reset"

# 자동 완료 / 수동 완료 / 타임아웃을 **따로** 센다. 체인이 11단계까지 간 것이
# 사람이 → 를 열 번 눌러서였다면 그것은 이 런이 내려던 결과가 아니다
# (사용자 결정 10/06: 두 모델 모두 수동키를 켜 두고 구분해 기록한다).
POLICY_OUTCOMES = ("complete", "manual", "timeout")
RESET_OUTCOMES = ("reached", "not_reached")

# `reason_detail.reset_end_reason` (executors.RESET_REASON_*). `not_reached` 는
# 세 가지 다른 사건의 **결과**라서, 구분 없이 표에 찍으면 사람이 눌러서 끊긴
# 램프와 정말 못 도달한 램프가 같은 칸에 들어간다.
RESET_END_REASON_LABEL = {
    "reached": "도달",
    "refused": "거부(움직이기 전)",
    "ceiling": "상한 초과",
    "manual_interrupt": "**사람이 끊음(→)**",
}


def read_events(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            print(
                f"[warn] {path}:{line_number} 을 JSON 으로 못 읽었다: {error}",
                file=sys.stderr,
            )
    return records


def read_basevel(path: Path | None) -> list[dict[str, float]]:
    """basevel.csv 를 읽는다. 없으면 빈 리스트 -- 베이스 열은 '-' 가 된다.

    없을 수 있는 경우가 실제로 있다: mock 체인(로봇이 없다), `LEROBOT_BASE_VEL_LOG`
    를 주지 않은 런, 그리고 `flush_base_vel_log` 가 `disconnect()` 에서 돌기 전에
    프로세스가 죽은 런.
    """
    if path is None or not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "t_mono": float(row["t_mono"]),
                        "meas_x_vel": float(row["meas_x_vel"]),
                        "meas_theta_vel": float(row["meas_theta_vel"]),
                        "cmd_x_vel": float(row["cmd_x_vel"]),
                        "cmd_theta_vel": float(row["cmd_theta_vel"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def integrate(
    rows: list[dict[str, float]], start: float | None, end: float | None
) -> dict[str, float] | None:
    """[start, end) 구간의 ∫x·∫θ (명령·실측). eval_base_stats.py:45-52 의 수식.

    Δt 는 그 행부터 다음 행까지의 시간이다 -- 베이스는 속도 명령을 **다음
    send_action 까지 유지**하므로, 느린 루프는 그만큼 더 돈다. 마지막 행의 Δt 는
    구간 내 중앙값으로 메운다(같은 스크립트의 규칙).
    """
    if not rows or start is None or end is None:
        return None
    window = [row for row in rows if start <= row["t_mono"] < end]
    if len(window) < 2:
        return None
    times = [row["t_mono"] for row in window]
    deltas = [b - a for a, b in zip(times, times[1:])]
    deltas.append(statistics.median(deltas))
    totals = {"cmd_x": 0.0, "meas_x": 0.0, "cmd_theta": 0.0, "meas_theta": 0.0}
    for row, delta in zip(window, deltas):
        totals["cmd_x"] += row["cmd_x_vel"] * delta
        totals["meas_x"] += row["meas_x_vel"] * delta
        totals["cmd_theta"] += row["cmd_theta_vel"] * delta
        totals["meas_theta"] += row["meas_theta_vel"] * delta
    totals["rows"] = float(len(window))
    return totals


def pair_stages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """stage_start/stage_end 를 짝지어 단계 레코드를 만든다.

    짝이 안 맞으면 경고하고 넘긴다 -- 보고서는 중단된 런에서도 나와야 한다.
    짝 검사 자체는 `aggregate.build_trial` 이 더 엄격하게 하고, 거기서 raise 하는
    것이 맞다(그쪽은 지표를 내고 이쪽은 사람이 읽는 표다).
    """
    pending: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = []
    for event in events:
        if event["event"] == "stage_start":
            if pending is not None:
                print(
                    f"[warn] stage_end 없는 stage_start: {pending.get('stage_id')!r}",
                    file=sys.stderr,
                )
            pending = event
        elif event["event"] == "stage_end":
            if pending is None:
                print(
                    f"[warn] stage_start 없는 stage_end: {event.get('stage_id')!r}",
                    file=sys.stderr,
                )
                continue
            if pending.get("stage_id") != event.get("stage_id"):
                print(
                    f"[warn] 짝이 안 맞는다: {pending.get('stage_id')!r} -> "
                    f"{event.get('stage_id')!r}",
                    file=sys.stderr,
                )
            stages.append({"start": pending, "end": event})
            pending = None
    if pending is not None:
        print(
            f"[warn] 마지막 단계가 끝나지 않았다: {pending.get('stage_id')!r} "
            "(프로세스가 그 안에서 죽었다)",
            file=sys.stderr,
        )
    return stages


def fmt(value: Any, spec: str = ".2f", dash: str = "-") -> str:
    if value is None:
        return dash
    if isinstance(value, float) and not math.isfinite(value):
        return dash
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def build_report(
    run_directory: Path, basevel_path: Path | None
) -> tuple[str, dict[str, Any]]:
    events = read_events(run_directory / "events.jsonl")
    rows = read_basevel(basevel_path)
    stages = pair_stages(events)

    trial_start = next((e for e in events if e["event"] == "trial_start"), {})
    trial_end = next(
        (e for e in reversed(events) if e["event"] == "trial_end"), {}
    )
    transitions = [e for e in events if e["event"] == "transition"]

    lines: list[str] = []
    run_id = trial_start.get("run_id") or run_directory.name
    lines.append(f"# 체인 회차 {run_id}")
    lines.append("")
    policies = trial_start.get("policies") or []
    checkpoint = policies[0] if policies else {}
    if not policies:
        # Reset-only (`chain.reset.only`): the ramps load no weights, so
        # trial_start carries no policy. Saying "없음" is the fact; four `?`
        # would read like a parse failure.
        lines.append(
            "- 체크포인트: **없음** (리셋 전용 — 가중치를 로드하지 않았다. "
            "`chain.reset.only` / `RESET_ONLY=1`)"
        )
    else:
        lines.append(
            f"- 체크포인트: `{checkpoint.get('policy_path', '?')}` "
            f"(state {checkpoint.get('state_dim', '?')}D → action "
            f"{checkpoint.get('action_dim', '?')}D, n_action_steps "
            f"{checkpoint.get('n_action_steps', '?')}, warm-up "
            f"{checkpoint.get('warmed_up', '?')})"
        )
    lines.append(
        f"- 데이터셋: `{trial_start.get('dataset_repo_id', '?')}` @ "
        f"{trial_start.get('fps', '?')} fps · 로봇 "
        f"`{trial_start.get('robot_type', '?')}` "
        f"(state {trial_start.get('robot_state_dim', '?')}, action "
        f"{trial_start.get('robot_action_dim', '?')})"
    )
    lines.append(
        f"- lerobot {trial_start.get('lerobot_version', '?')} · stage_runner "
        f"{trial_start.get('stage_runner_version', '?')} · config v"
        f"{trial_start.get('config_version', '?')}"
    )

    reason = trial_end.get("reason", "trial_end 없음 (프로세스가 먼저 죽었다)")
    lines.append(
        f"- 결과: **{reason}** · completed={trial_end.get('completed')} · "
        f"저장={trial_end.get('episode_saved')} · "
        f"{trial_end.get('stage_count', len(stages))}/"
        f"{trial_end.get('stages_configured', '?')} 단계 · "
        f"{trial_end.get('buffered_frames', '?')} 프레임 · "
        f"{fmt(trial_end.get('elapsed_s'), '.1f')} s"
    )
    if trial_end.get("chain_failed"):
        lines.append(
            f"- **체인이 끊긴 단계: `{trial_end.get('chain_failed_stage_id')}`** "
            "(재시도 없음 — 이후 단계는 돌지 않았다)"
        )
    lines.append("")

    # ---- 정책 단계 표 ----
    lines.append("## 정책 단계")
    lines.append("")
    if not any(
        record["start"].get("kind") == STAGE_KIND_POLICY for record in stages
    ):
        lines.append("없음 — 리셋 전용 회차다 (bring-up ②·③).")
        lines.append("")
    lines.append(
        "| 단계 | 종료 | 출발 s | 경과 s (p10/p90) | Hz | p 끝값·유지 s | 정지 s | "
        "NN rad | ∫θ 명령/실측 ° | ∫x 명령/실측 m | clamped |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    counts = {"policy": {}, "reset": {}}
    never_departed: list[str] = []
    for record in stages:
        start, end = record["start"], record["end"]
        if start.get("kind") != STAGE_KIND_POLICY:
            continue
        detail = end.get("reason_detail") or {}
        terminator = end.get("terminator", "?")
        counts["policy"][terminator] = counts["policy"].get(terminator, 0) + 1
        base = integrate(rows, start.get("t_mono"), end.get("t_mono"))
        # `departure_unknown` 은 **안 떠났음이 아니라 모름**이다 — 감시자가 팔을
        # 못 읽은 단계에서 0.000 rad 를 측정값처럼 세면 볼 곳을 틀리게 가리킨다.
        if detail.get("departed") is False and not detail.get("departure_unknown"):
            never_departed.append(str(start.get("stage_id")))
        lines.append(
            "| {stage} | {term} | {departed} | {elapsed} ({p10}/{p90}) | {hz} | {p} | "
            "{stall} | {nn} | {theta} | {x} | {clamped} |".format(
                stage=f"`{start.get('stage_id')}`",
                term=f"**{terminator}**",
                # 「언제 시작 장면을 떠났나」. `✗` 는 **한 번도 떠나지 않았다** --
                # 그 단계는 어려워서 실패한 것이 아니라 시작하지 않았고, 그러면
                # 단계 자체에 대해 측정된 것이 없다. `?` 는 **모름** (감시자가
                # 눈이 멀어 팔을 못 읽었다 — 0.000 rad 는 측정값이 아니다).
                # 옛 회차의 events.jsonl 에는 이 키가 없어서 `-` 가 된다
                # (없음 ≠ 안 떠났음).
                departed=(
                    "?"
                    if detail.get("departure_unknown")
                    else "✗"
                    if detail.get("departed") is False
                    else fmt(detail.get("departed_s"), ".1f")
                ),
                elapsed=fmt(end.get("elapsed_s"), ".1f"),
                p10=fmt(detail.get("p10_s"), ".1f"),
                p90=fmt(detail.get("p90_s"), ".1f"),
                hz=fmt(end.get("hertz"), ".1f"),
                p=(
                    f"{fmt(detail.get('p_last'), '.3f')}·"
                    f"{fmt(detail.get('p_hold_s'), '.2f')}"
                    if detail.get("p_last") is not None
                    else "-"
                ),
                stall=fmt(detail.get("stall_s"), ".2f"),
                nn=fmt(detail.get("nn_dist"), ".3f"),
                theta=(
                    f"{fmt(math.degrees(base['cmd_theta']), '+.1f')}/"
                    f"{fmt(math.degrees(base['meas_theta']), '+.1f')}"
                    if base
                    else "-"
                ),
                x=(
                    f"{fmt(base['cmd_x'], '+.3f')}/{fmt(base['meas_x'], '+.3f')}"
                    if base
                    else "-"
                ),
                clamped=fmt(detail.get("clamped_ticks"), "d"),
            )
        )
    lines.append("")

    # ---- 리셋 표 ----
    lines.append("## 경계 리셋")
    lines.append("")
    lines.append(
        "| 리셋 | 종료 | 사유 | T s | Δmax rad (상한) | 보폭 상한 rad | 도달 오차 rad "
        "(tol) | 경과 s (상한) | Hz | 프레임 | clamped |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    interrupted: list[str] = []
    for record in stages:
        start, end = record["start"], record["end"]
        if start.get("kind") != STAGE_KIND_RESET:
            continue
        detail = end.get("reason_detail") or {}
        terminator = end.get("terminator", "?")
        counts["reset"][terminator] = counts["reset"].get(terminator, 0) + 1
        initial = start.get("stage_id", "").startswith("reset_pre_")
        end_reason = detail.get("reset_end_reason")
        if end_reason == "manual_interrupt":
            interrupted.append(str(start.get("stage_id")))
        lines.append(
            "| {label} | {term} | {why} | {T} | {dmax} ({limit}) | {step} | "
            "{err} ({tol}) | {elapsed} ({ceiling}) | {hz} | {frames} | "
            "{clamped} |".format(
                label=f"`{start.get('stage_id')}`" + (" (최초)" if initial else ""),
                term=f"**{terminator}**",
                # `not_reached` 는 세 사건의 결과다 — 거부(움직이기 전) ·
                # 상한 초과 · 사람이 `→` 로 끊음. 셋을 섞으면 잘 돌던 램프를
                # 실패한 램프로 읽는다.
                why=RESET_END_REASON_LABEL.get(end_reason, end_reason or "-"),
                T=fmt(detail.get("reset_T"), ".2f"),
                limit=fmt(detail.get("reset_jump_limit_rad"), ".2f"),
                dmax=fmt(detail.get("reset_dmax"), ".3f"),
                step=fmt(detail.get("reset_max_step_rad"), ".4f"),
                err=fmt(detail.get("reach_err"), ".4f"),
                tol=fmt(detail.get("reset_tol_rad"), ".3f"),
                elapsed=fmt(end.get("elapsed_s"), ".1f"),
                ceiling=fmt(detail.get("reset_ceiling_s"), ".1f"),
                # 리셋 구간 Hz 는 **읽어야 하는 수치**다 — 10/07 에 여기가
                # 11.9~13.1 Hz 였고(목표 21), 원인은 리셋 번들의 CPU 영상
                # 변환이었다. `⚠` 는 그 생략이 **안 걸렸다**는 뜻이다
                # (`reason_detail.predict_path`): 그러면 낮은 Hz 는 램프 탓이
                # 아니라 변환 탓이다. 옛 회차의 events.jsonl 에는 키가 없어서
                # 표시가 붙지 않고(`None`), `not_called` 도 붙지 않는다 —
                # 첫 틱 전에 끊긴 리셋은 **아무것도 추론하지 않은** 것이라
                # 「생략이 안 걸렸다」와 다르다 (없음 ≠ 안 걸림).
                hz=fmt(end.get("hertz"), ".1f")
                + (
                    "⚠"
                    if detail.get("predict_path")
                    not in (None, "state_only", "not_called")
                    else ""
                ),
                frames=fmt(end.get("frames"), "d"),
                clamped=fmt(detail.get("clamped_ticks"), "d"),
            )
        )
    lines.append("")

    # ---- 집계 ----
    lines.append("## 집계")
    lines.append("")
    policy_total = sum(counts["policy"].values())
    reset_total = sum(counts["reset"].values())
    policy_line = " · ".join(
        f"{name} {counts['policy'].get(name, 0)}"
        for name in POLICY_OUTCOMES
        if counts["policy"].get(name)
    ) or "없음"
    other_policy = {
        k: v for k, v in counts["policy"].items() if k not in POLICY_OUTCOMES
    }
    reset_line = " · ".join(
        f"{name} {counts['reset'].get(name, 0)}"
        for name in RESET_OUTCOMES
        if counts["reset"].get(name)
    ) or "없음"
    other_reset = {
        k: v for k, v in counts["reset"].items() if k not in RESET_OUTCOMES
    }
    lines.append(f"- 정책 단계 {policy_total}개: {policy_line}")
    if other_policy:
        lines.append(f"  - 그 밖: {other_policy} (중단·오류)")
    boundary = sum(
        1
        for record in stages
        if record["start"].get("kind") == STAGE_KIND_RESET
        and not record["start"].get("stage_id", "").startswith("reset_pre_")
    )
    lines.append(
        f"- 리셋 {reset_total}개 (경계 {boundary} + 최초 "
        f"{reset_total - boundary}): {reset_line}"
    )
    if other_reset:
        lines.append(f"  - 그 밖: {other_reset} (중단·오류)")
    lines.append(
        "- **자동 완료 {auto} / 수동 완료 {manual}**: 수동이 있으면 그만큼은 "
        "모델이 끝을 스스로 알리지 못한 것이다 (사용자 결정 10/06 — 두 모델 모두 "
        "`→` 를 켜 두고 따로 센다)".format(
            auto=counts["policy"].get("complete", 0),
            manual=counts["policy"].get("manual", 0),
        )
    )
    if never_departed:
        lines.append(
            f"- **출발하지 않은 단계 {len(never_departed)}개: {never_departed}** — "
            "실측 팔이 지정 시작 자세를 떠나지도, 베이스가 돌지도 않았다. 그 "
            "단계의 난이도에 대해 측정된 것은 **없다**. 볼 곳은 정책이 아니라 "
            "시작 장면이다: 바로 앞 리셋의 도달 오차 · 원핫 인덱스 · 카메라"
        )
    if interrupted:
        lines.append(
            f"- **리셋 중 `→` 로 끊긴 램프 {len(interrupted)}개: {interrupted}** — "
            "`→` 는 정책 단계 전용이다(리셋은 스스로 도달을 판정한다). 끊긴 "
            "램프는 `not_reached` 이고 그것은 체인 실패다 — 팔이 다음 단계 시작 "
            "자세에 **없다**"
        )
    # `primary` 가 아닌 **모든** 경로가 실패다. `primary+direct_failed` 는 홀드
    # 액션은 갔는데 확인용 `base.set_cmd_vel(0,0)` 이 거부된 경계이고, 그때
    # 베이스가 멈췄다는 증거는 **없다** — `send_action` 은 베이스 쓰기가 실패해도
    # 예외를 내지 않는다(mobileai.py:547-558). `== "failed"` 만 보던 옛 판은
    # 그 경계를 조용히 「전부 primary」 로 셌다.
    failed_stops = [
        t
        for t in transitions
        if (t.get("stop_base_path") or "failed") != "primary"
    ]
    lines.append(
        f"- 경계 {len(transitions)}곳의 베이스 정지: "
        + (
            f"**{len(failed_stops)}곳이 primary 아님 "
            f"({sorted({str(t.get('stop_base_path')) for t in failed_stops})}) — "
            "로봇을 확인하라**"
            if failed_stops
            else "전부 primary (확인용 set_cmd_vel 까지 수락됨)"
        )
    )
    clamped = [
        (record["start"].get("stage_id"), (record["end"].get("reason_detail") or {}).get("clamped_ticks"))
        for record in stages
    ]
    nonzero = [(name, n) for name, n in clamped if n]
    lines.append(
        "- max_relative_target 클램프: "
        + (
            f"{nonzero} — **리셋에서 0 이 아니면 램프가 틀렸다** (클램프는 안전망이다)"
            if nonzero
            else ("전 단계 0" if any(n == 0 for _, n in clamped) else "미측정 (로봇 밖)")
        )
    )
    if not rows:
        lines.append(
            "- 베이스 적분: **없음** (basevel.csv 가 없다 — `LEROBOT_BASE_VEL_LOG` "
            "를 주지 않았거나, mock 런이거나, `disconnect()` 의 flush 전에 죽었다)"
        )
    else:
        lines.append(f"- basevel.csv {len(rows)} 행")
    lines.append("")
    lines.append(
        "> 성공·실패 **판정은 여기 없다.** 영상과 사람이 적은 표에서 오고 "
        "`~/eval_logs/eval_chain_results.csv` 로 들어간다. 이 보고서는 러너가 "
        "측정한 것만 담는다."
    )
    lines.append("")

    summary = {
        "run_id": run_id,
        "reason": reason,
        "policy": counts["policy"],
        "reset": counts["reset"],
        "never_departed": never_departed,
        "reset_manual_interrupt": interrupted,
        "failed_base_stops": len(failed_stops),
        "basevel_rows": len(rows),
    }
    return "\n".join(lines), summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="outputs/stage_runner/<회차>")
    parser.add_argument(
        "--basevel",
        type=Path,
        default=None,
        help="~/eval_logs/<회차>.basevel.csv (없으면 베이스 열은 '-')",
    )
    parser.add_argument("--out", type=Path, default=None, help="보고서를 쓸 파일")
    parser.add_argument("--json", type=Path, default=None, help="집계를 JSON 으로")
    arguments = parser.parse_args(argv)

    events_path = arguments.run_directory / "events.jsonl"
    if not events_path.exists():
        print(f"!! {events_path} 가 없다", file=sys.stderr)
        return 2

    basevel = arguments.basevel
    if basevel is not None:
        basevel = basevel.expanduser()
    report, summary = build_report(arguments.run_directory, basevel)
    print(report)
    if arguments.out:
        arguments.out.expanduser().write_text(report + "\n", encoding="utf-8")
        print(f"\n-- {arguments.out} 에 썼다", file=sys.stderr)
    if arguments.json:
        arguments.json.expanduser().write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
