# 2026-09-22 SmolVLA 오프라인 평가 원본 (ibomcom)

정리된 기록: [../../offline_eval_2026-09-22.md](../../offline_eval_2026-09-22.md)

| 경로 | 내용 |
|---|---|
| `chunks/<tag>.json` | 예측 실행마다 명령줄·체크포인트 경로·에피소드·설정·GPU. 짝이 되는 `<tag>.npz`(예측 청크, 수백 MB)는 git 에 올리지 않는다 |
| `sim/*.jsonl` | `simulate.py` 결과 (teacher-forced 실행 스케줄) |
| `closedloop/*.jsonl` | `closedloop_eval.py` 결과. `*_perfect.jsonl` 은 같은 스케줄의 완벽 정책 기준선. 에피소드별 행이 `per_episode` 에 있다 |
| `openloop_all.jsonl`, `openloop_table.md` | `summarize.py` 로 모은 열린 루프 지표 전체 |
| `logs/` | 각 실행의 표준 출력. `_status.txt` 에 종료 코드 |

실행은 브랜치 `offline-eval-0922` 의 워크트리에서 커밋 전 코드로 했다. json 의 `git_sha`(15f667f)는 그 기준 커밋이다.

## 다시 만들기

GPU 는 UUID 로 지정한다. `scripts/offline_eval/run_*.sh` 가 이번에 돌린 순서 그대로다.

| 스크립트 | 만드는 것 | 대략 시간 (3090 한 장) |
|---|---|---|
| `run_phase1.sh` | 80k 모든 프레임(task02·task06), 계보 3개 | 60분 |
| `run_phase2.sh` | GIST 판, 프롬프트, state 지연 | 40분 |
| `run_phase3.sh` | medoid·K=4 noise 1.0·noise 1.2 | 60분 |
| `run_phase4.sh` | 베이스 속도 state 가리기 | 10분 |
| `run_closedloop.sh`, `run_closedloop36.sh` | 재생 폐루프 | 60분, 40분 (3장) |
| `run_closedloop_offset.sh`, `run_closedloop_drift.sh` | 어깨 보정 | 20분씩 |

ACT 기준선은 `act_chunks.py --policy kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator_260907 --dataset kiroaiseoul/task06_pickup_beaker_and_move_to_refrigerator --stride 3 --tag act60k_t06`.
GIST 판은 허브에 codebase 버전 태그가 없어서 `fetch_episodes.py` 로 필요한 파일만 받은 뒤 `--root` 로 읽었다.
