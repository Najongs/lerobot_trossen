# CLAUDE.md

> **세션 시작 확인** — 이 레포는 Trossen PC1 공용 계정(`trossen-ai`)에 있다. Claude 는
> `source ~/NAJY/.tools/env.sh && claude` 로 연다. `echo $CLAUDE_CONFIG_DIR` 가
> `/home/trossen-ai/NAJY/.claude-najy` 가 아니면 공용 `~/.claude` 로 뜬 것이다 —
> 전역 규칙·MCP(vaultgraph·arxiv 등)·레포 메모리가 안 보인다. 세션을 닫고 다시 열어라.

> **DGX_1(`/home/kiro-ai/NAJY/lerobot_trossen`)에서 열었다면** 위 머리말은 로봇 PC 기준이다 — 먼저
> `docs/HANDOFF_najy_1009.md`(**새 세션 진입점** — 목표·확정 교훈·돌아가는 런·후보 체크포인트·다음 순서) → `docs/najy_overview.md`(현황 정본) →
> `docs/HANDOFF_najy_1006.md`(DGX 경로·오프라인 도구·주의사항).
> DGX 에서는 이 레포 패키지를 import 하지 마라(로봇 SDK 를 끌어온다), push 는 `najongs` 원격(fork)으로.

Trossen Mobile AI(WidowX AI 팔 4대 + SLATE 베이스 + RealSense 3대)용 lerobot 플러그인 fork.
11단계 실험실 작업(튜브→비커→냉장고→선반)을 ACT 로 녹화·학습·실기 eval 한다.
명령·인자의 정본은 `README.md`. 학습 쪽 배경은 `docs/` 의 NAJY 문서들(아래 「실험」).

## 실행 규약
- 환경: **uv + Python 3.11** (`.python-version`). conda 안 씀. 레포 루트에서 `uv run …` (README §0, 7429a76)
- 실측 `.venv`: lerobot 0.4.4 · torch 2.10.0+cu128 · transformers 4.53.3(fork) · trossen_slate 0.0.3. `--extra pi0` 없이 sync 된 상태
- torch 는 2.8~2.10 cu128 고정 — 올리지 마라 (dc52029, 4d8963a)
- 환경변수 전체 표: README §Environment Variables. 기본 ON: `LEROBOT_FAST_OBS` · `LEROBOT_LOOP_HZ_LOG` · `LEROBOT_BASE_SERIAL_REARM`
- 진입점: 녹화 `uv run lerobot-record` · 학습 `uv run accelerate launch -m lerobot.scripts.lerobot_train` ·
  다단계 ACT eval `scripts/eval_najy.sh <M1|M2|M3|repo> <단계> <exec> [ep]` (`DRY_RUN=1` 지원) · SmolVLA `scripts/eval_smolvla.sh`
- **실기 정본 체크아웃은 이 레포(`~/NAJY/lerobot_trossen`)다** — 10-02 eval_najy 런이 전부 여기서 돌았다(`~/eval_logs/1002_*`).
  홈의 `~/lerobot_trossen` 은 팀 공용 체크아웃(kiro-ai-division, 원핫 패치 없음)이라 손대지 않는다. README 의 `cd ~/lerobot_trossen` 은 팀 기준 표기

## 실기 안전 — 로봇 코드는 직접 실행하지 않는다
`lerobot-record`·`lerobot-teleoperate`·`eval_*.sh`·`measure_base.py`(dry-run 제외) 는 로봇을 움직이거나 포트를 연다.
**명령을 만들어 사용자에게 넘긴다.** 상세는 전역 `rules/robot-safety.md`.
- 주소: follower 좌 192.168.1.5 / 우 .4, leader 좌 .3 / 우 .2. 베이스는 시리얼(Modbus) (README:35-41)
- 안전 상수 (`packages/lerobot_robot_trossen/src/lerobot_robot_trossen/`):
  `velocity_safety_factor=0.4` (config_bi_widowxai_follower.py:31) — 0.5·0.8 은 실기 트립 (f3e66e9), 올리지 마라 ·
  베이스 명령 ±1.0 클램프·NaN→0 (mobileai.py:159-179) · `estop_check=True` (config_mobileai.py:29)
- dry-run 이 끄는 범위: `eval_najy.sh DRY_RUN=1` 은 로봇 무접촉이지만 `~/eval_logs` 생성·HF 다운로드·`uv run` 2회는 한다.
  `lerobot-record`·로봇 클래스엔 dry-run 이 없다. `enable_base_motor_torque=false` 는 dry-run 이 아니다 — 팔은 움직인다
- 베이스 시리얼 포트 경합: `measure_base rate` 등 오프라인 도구가 포트를 잡은 직후 런에서 베이스 통신이 전부 실패한 사례 (smolvla_inference_experiments.md:190)

## 절대 깨면 안 되는 것
- `--policy.type` 을 `--policy.path` 없이 주면 **랜덤 가중치 정책이 경고 없이 로봇을 구동**한다 (README:340)
- 27D/20D 다단계 ACT 는 `LEROBOT_TASK_ONEHOT=<i>/<K>` (**1-based**) 없이는 죽는다. 로그에 `installed`·`stage i/K active` 가 없으면 중단.
  `lerobot-record` 진입점만 패치된다 (README §Stage One-Hot). 학습 매니페스트는 0-based(`[0,11]`) — 패치가 변환한다
- 관측 차원 14/16/16+K. 기본 `include_base_in_state=false`(베이스 칸 0). 9/16 이전 16D 체크포인트만 `true` (a015c68)
- 실데이터 16D = `[left7, right7, x.vel, theta.vel]` (베이스가 뒤). sim 텔레옵 데이터만 베이스가 앞 — 섞으면 차원만 맞고 조용히 틀린다

## 데이터 보호
- 데이터셋: `~/.cache/huggingface/lerobot/<repo-id>` (실측 `kiroaiseoul/` 169개). 원본 녹화는 유일본일 수 있다 — 지우기 전 확인
- **`push_to_hub` 기본 true.** 스모크의 `false` 를 본학습으로 옮기지 말 것, warm start 에 부모 repo_id 재사용 금지(덮어씀),
  edit-dataset 은 `--new_repo_id` 필수 (README:172, 324, 380-381, 433)
- `eval_smolvla.sh:48` 은 매 실행 HF 캐시의 eval 데이터셋을 `rm -rf` 한다
- 재생성 가능: `outputs/`(gitignore), `~/eval_logs/` 는 결과 원본이니 옮기기 전 지우지 마라

## 실험
- **전체 현황 한 장 — `docs/najy_overview.md`** (학습 결론·실기 현황·다음·무엇이 어디 있나, 1·2호기 통합)
- **다단계 ACT 실기 Eval 을 이어 갈 때는 `docs/eval_najy_session_guide.md` 부터** — 현재 위치 찾기, Claude 가 직접 할 것/사람에게 넘길 것, 회차 진행·기록 형식.
  순서표·명령 정본: `docs/eval_najy.md`. 지난 결과: `docs/eval_najy_results_<MMDD>.md` (가장 최근 것의 「다음」 절이 현재 위치). 실기 exec 는 30 으로 확정(10-02). 학습·오프라인 판정 배경: `docs/act_checkpoints_najy.md` · `docs/data_layout_traps.md` · `docs/offline_eval_findings_act.md`
- SmolVLA 권장 설정 정본: `docs/offline_eval_2026-09-22.md`. 런 로그는 `docs/run_logs/<날짜>_*` (`*.log` 도 추적됨)
- 학습 정본은 private 레포 `Najongs/trossen-ai-simulation` (`docs/mobile_base_investigation.md` §82~87, `docs/real_robot_eval.md`)
- **낡은 것 — 근거로 쓰지 마라**: 「ACT temporal ensemble 필수」(1aacc90 → aadcb08 철회) · smolvla_inference_experiments.md 「현재 쓸 수 있는 구성」(offline_eval §5 가 뒤집음) ·
  implementation_code.sh 상단 SmolVLA 명령 · `Start.txt`(개인 메모, 정본 아님)

## 알려진 함정
- eval repo 는 `eval_` 접두사 필수, 같은 이름이면 FileExistsError. `| tail` 을 붙이면 실패가 exit 0 이 된다 (README:344, 357)
- `--dataset.fps=21` 을 빠뜨리면 serial rearm 이 꺼진다. eval 은 `enable_base_motor_torque=true` (README:343, 350)
- `n_action_steps>1` 이면 temporal ensemble 금지 — 루프 20% 느려지고 과회전 1.29× (aa30006)
- `Joint 0 … NaN`: 손상된 normalizer stats 또는 fp16 오버플로 — 둘을 구분하라 (README:359, a9c2a8a)
- `--optimizer.lr` 은 무시된다. `--robot.cameras` 바깥 따옴표는 홑따옴표 (README:317, 386)
- 과회전 배수 ≈ 녹화 루프 Hz ÷ eval 루프 Hz. eval 전 녹화 Hz·`measure_base.py` 로 기준을 잡는다
- 다른 사람 경로 하드코딩: `/home/trossen-ai/daehee/models/…` (eval_smolvla.sh:24, implementation_code.sh:54)
- 학습은 lerobot 0.4.1, 로봇 PC 는 0.4.4 — 전처리기 호환은 문서 기록 없음

## git
- remote: 이 체크아웃은 `origin = Najongs/lerobot_trossen`(HTTPS) 하나. **public 레포**다 — 토큰·비밀번호를 넣지 마라
- `main` 에 직접 커밋·push (레포 메모리 commit-straight-to-main). upstream 은 kiro-ai-division [추정 — 이 체크아웃엔 remote 없음]
- 커밋 메시지: 영어 `Feat:/Fix:/Docs:` 와 한국어 `feat(eval):/docs:` 혼재 — 본인 커밋은 한국어 `타입(범위): 제목` 관례
- 커밋 전 `git config user.name/user.email` 확인 (공용 계정)
- `.pre-commit-config.yaml` 에 `no-commit-to-branch` 가 있으나 훅은 미설치
- `git add` 는 파일 명시. `Start.txt` 는 다른 사람도 고치는 메모장 — 내 커밋에 휩쓸지 마라
