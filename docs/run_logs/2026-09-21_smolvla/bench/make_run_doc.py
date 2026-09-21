import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))

S = sys.argv[1]
idx = open(f"{S}/index.md").read()
table = idx[: idx.index("\n\n- T1")].strip()
files = [l.split("\t") for l in idx[idx.index("FILES") + 6 :].strip().splitlines()]

HEAD = '''# SmolVLA 로봇 런 기록 (2026-09-21 ~ 09-22)

[smolvla_inference_experiments.md](smolvla_inference_experiments.md)의 부록이다. 남아 있는 로봇 런 로그 40개에서 실행 파라미터와 측정값을 뽑아 표로 만들었다. 09-21 23:23 이후의 파라미터 스윕(R10~R40)은 이 표가 유일한 기록이다.

- 원본 로그와 벤치마크 스크립트는 `docs/run_logs/2026-09-21_smolvla/`에 넣어 두었다. `.gitignore`의 `*.log`에 이 폴더만 예외로 걸어서 git에 올라간다. 표는 그 폴더의 `bench/index_runs.py`로 다시 만들 수 있다.
- 09-21 19:01 ~ 21:09의 초기 런(E1, E3~E5, E7)은 `tee` 없이 돌려서 로그가 없다. 수치는 실험 기록 문서에만 있다.
- `smolvla_every25_s50.log`, `smolvla_noTA_bf16_s20.log`, `smolvla_ns05_k4_s50.log`는 같은 파일명으로 여러 번 덮어써서 마지막 런만 남았다. 표의 R1~R3은 파일명이 아니라 로그 내용 기준이다.

## 로그에 남는 것과 남지 않는 것

| 남는 것 | 어디에 |
|---|---|
| lerobot 설정 전체(`num_steps`, `single_task`, 모델 경로, 카메라, 데이터셋 인자) | 시작 직후의 설정 덤프 |
| `--ensemble.*` 값 | `temporal ensembling 활성화 (…)`, `샘플링 조정: …`, `commit=N: …`, `청크 병합: …` 줄. 동기 모드에서는 `amp` 값이 찍히지 않는다 |
| 추론 횟수·평균·최악, 도착 지연 | 종료 직전의 `추론 N회 …` 줄. `tee -i`가 아니면 Ctrl+C 때 잘린다 |
| 제어 루프 주파수, 구간별 시간 | `Control loop rate …` 줄 (30프레임마다) |
| 클램프, base 통신 실패 | WARNING 줄 |

**동작의 품질은 로그에 남지 않는다.** 부드러웠는지, 과제를 끝까지 했는지는 어떤 로그 줄로도 알 수 없다. 간접 지표는 정책 구간 클램프 횟수 하나뿐이고, 이것도 "급한 목표가 몇 번 나왔나"이지 성공 여부가 아니다. 이번 40개 런에서는 에피소드가 한 번도 저장되지 않아서(전부 왼쪽 화살표나 Ctrl+C로 종료) 액션 궤적으로 되짚어 볼 수도 없다. 그래서 표 맨 끝에 **관찰** 열을 비워 두었다. 기억나는 런은 직접 채워 넣기 바란다.

대화 중에 확인된 관찰은 다음이 전부다.

| 런 | 관찰 |
|---|---|
| R6 (연속 TA, coeff 0.01) | GPU가 계속 돌았다. 클램프 0회. 동작 평가는 듣지 못했다 |
| R7 (`merge=latest`, commit 없음) | "뚝딱뚝딱" 끊어지며 움직였다. 클램프 135회 |
| R9 (`every=40`) | GPU가 쉰다는 지적을 받았다 |
| R40 (`commit=50`, K=7, T4) | 맞는 방향으로는 가는데 끝까지 수행하지 못했다 |
| 로그 없음 (동기, `num_steps=50`) | 20스텝보다 확실히 부드러웠다 |
| 로그 없음 (동기, `every=25`, 50스텝) | 멈췄다 움직였다를 반복했고 손이 나가지 못했다 |

## 앞으로 기록하는 법

1. 로그를 `/tmp`가 아니라 레포 안의 `docs/run_logs/<날짜>_<이름>/`에 받는다. 이 경로의 `*.log`는 git에 올라간다.
   ```bash
   mkdir -p docs/run_logs/$(date +%F)_smolvla
   ... 2>&1 | tee -i docs/run_logs/$(date +%F)_smolvla/run_$(date +%H%M%S).log
   ```
2. 런이 끝나면 관찰을 그 로그에 한 줄 붙인다. 파라미터와 관찰이 같은 파일에 남는다.
   ```bash
   echo "관찰: 집기 성공, 흔들기에서 떨림, base 과회전 없음" >> docs/run_logs/.../run_HHMMSS.log
   ```
3. 비교할 런은 **오른쪽 화살표**로 끝내서 에피소드를 저장한다. 저장된 에피소드가 있어야 청크 경계의 점프 크기나 시연과의 차이를 숫자로 비교할 수 있다.

## 과제(프롬프트)

- T1: "Grab and shake the filled beaker, then move to the front of the fridge"
- T2: "Grab and shake the filled beaker, t fridge" — 명령줄 편집 중에 깨진 문장이다. SmolVLA는 프롬프트가 정책 입력이므로 R10~R14의 동작은 다른 런과 비교하면 안 된다
- T3: "Move to the tube rack"
- T4: "Pick up the two tubes from the rack with both hands"

## 런 표

- 모델은 전부 `smolvla_190k_30k_aug_80k_kirogist`, `include_base_in_state=true`다.
- "정책 구간 클램프"는 `Recording episode` 이후에 찍힌 `max_relative_target` 클램프 수다. 리셋 구간의 클램프는 리더 암에 맞춰지는 순간이라 뺐다. 왼쪽 화살표로 다시 시작한 시도들이 한 런에 합쳐져 있다.
- "정책 구간 초"는 정책이 로봇을 몬 시간의 합이다. 짧은 런(20초 미만)의 수치는 표본이 작다.
- `merge` 열은 `commit`이 없으면 도착한 청크를 합치는 방식이고, `commit`이 있으면 그 구간에 도착한 청크를 처리하는 방식이다.
- "왼쪽 화살표 N회"는 키 입력 메시지 수다. 키를 누르고 있으면 반복해서 찍히므로 시도 횟수와 같지 않다.

'''

TAIL = '''

눈에 띄는 점은 다음과 같다. 전부 로그 수치에서만 읽은 것이고 동작 품질과는 별개다.

- R4는 base 통신이 런 내내 실패해서 제어 루프가 29.9 Hz로 돌았다. 폐기한다.
- 프로세스 워커로 바꾼 뒤(R4 이후) 제어 루프는 `samples`가 4 이하이면 19.6~21.0 Hz로 일정하다. `samples=7`(R37, R40)에서는 18.5 Hz로 떨어졌다. 워커와 제어 스레드가 같은 GPU를 나눠 쓰기 때문으로 보이는데 확인하지는 않았다.
- 도착 지연은 추론 시간을 따라간다. 10스텝·K≤2에서 3스텝, 25스텝·K=4에서 6스텝이다.
- 같은 구성이어도 정책 구간 클램프 수가 런마다 크게 다르다(예: R23 0회, R25 5회, R26 19회). 시작 자세와 과제 진행 정도가 런마다 달라서, 클램프 수만으로 구성의 우열을 가리기는 어렵다.
- R22(`commit=50`, coeff 0, K=1)는 클램프가 80회로 튄다. `commit=50`은 실제로는 42~43스텝에서 버퍼가 말라 갈아타는 구성이다(실험 기록 E19).

## 로그 파일 대응

| 런 | 파일 |
|---|---|
'''

doc = HEAD + table + TAIL + "\n".join(f"| {r} | `{f}` |" for r, f in files) + "\n"
out = os.path.join(REPO, "docs/smolvla_run_index.md")
open(out, "w").write(doc)
print("written", out, len(doc.splitlines()), "lines")

p = os.path.join(REPO, "docs/smolvla_inference_experiments.md")
s = open(p).read()
old = "- 오프라인 벤치마크 스크립트는 `/tmp/smolbench/`에 있다. `/tmp`라서 재부팅하면 사라진다."
new = ("- 로봇 런별 파라미터와 측정값은 [smolvla_run_index.md](smolvla_run_index.md)에 표로 정리했다. 원본 로그와 오프라인 "
       "벤치마크 스크립트는 `docs/run_logs/2026-09-21_smolvla/`에 복사해 두었다.")
if old in s:
    open(p, "w").write(s.replace(old, new))
    print("linked from experiments doc")
else:
    print("link line already replaced" if new in s else "WARNING: link line not found")
