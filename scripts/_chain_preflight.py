#!/usr/bin/env python3
"""`scripts/eval_chain.sh` 의 2단계: 체크포인트 ↔ 체인 YAML ↔ 단계 파라미터 대조.

    uv run python scripts/_chain_preflight.py <POLICY 디렉터리> <체인 YAML> <stage_params.json>

로봇에 연결하기 전에, 그리고 수백 MB 짜리 safetensors 를 받기 전에 막는다
(`config.json` 만 읽는다). `scripts/eval_najy.sh:86-104` 의 폭 검사 블록을
체인용으로 옮긴 것이고, 다른 점은 셋이다:

- action 폭 **16(진행도 없음)과 17(progress)을 모두** 허용하고 어느 쪽인지 찍는다
- YAML 의 `chain.model.has_progress`·`onehot_k`·`n_action_steps` 와 대조한다.
  `has_progress` 는 **명시 필수**다 — 폭과 키가 **둘 다** 맞아야 통과한다
  (한쪽만이면 거부). 「폭에서 알아서 읽는다」 는 러너가 그 사실을 설정에 남기지
  않는다는 뜻이고, 그 사실이 녹화 데이터셋의 action 칸 수(=정규화기 폭)를 정한다
- 단계 파라미터를 **러너와 같은 로더**로 검증한다 (`chain_params`, stdlib only)

**YAML 은 파싱한다 — grep 하지 않는다.** 셸 안의 `grep -E '^\\s*fps:' | head -1`
은 카메라 블록의 `fps: 30` 을 먼저 집어 `dataset.fps: 21` 에 닿지 못한다. 이
스크립트가 별도 파일인 이유가 그것이다: 들여쓰기로 중첩을 표현하는 파일에서 키
이름만 보는 것은 같은 이름이 두 블록에 있는 순간 틀리고, `bash -n` 으로는 안
잡힌다.

표준 출력은 셸이 다시 읽는다 (`ONEHOT_K=` 줄이 있으면 원핫 패치를 확인한다).
사람이 읽을 줄은 표준 오류로 간다.

**로봇 SDK 를 import 하지 않는다** — `yaml`, `json`, 그리고 stdlib-only 인
`stage_runner.chain_params` 뿐이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "stage_runner" / "src"))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    policy_dir, yaml_path, params_path = argv

    import yaml

    from stage_runner.chain_params import ChainParamsError, load_chain_params

    config_path = Path(policy_dir) / "config.json"
    if not config_path.exists():
        print(f"!! {config_path} 가 없다 — 체크포인트 경로를 확인하라", file=sys.stderr)
        return 3
    checkpoint = json.loads(config_path.read_text(encoding="utf-8"))
    document = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    chain = document.get("chain") or {}
    model = chain.get("model") or {}
    dataset = document.get("dataset") or {}

    problems: list[str] = []

    try:
        state = int(checkpoint["input_features"]["observation.state"]["shape"][0])
        action = int(checkpoint["output_features"]["action"]["shape"][0])
        chunk = int(checkpoint["chunk_size"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        print(
            f"!! {config_path} 에서 폭·chunk 를 못 읽었다: {error!r}", file=sys.stderr
        )
        return 3

    # temporal ensemble: n_action_steps>1 과 같이 못 쓴다 (aa30006 — 루프 20%
    # 느려지고 과회전 1.29배).
    #
    # 여기는 **무조건** 거부한다 — 러너 안의 같은 게이트
    # (`preflight.temporal_ensemble_problem`) 보다 엄격하다. 그쪽은 실행
    # `n_action_steps > 1` 일 때만 막는다(ACT 가 막는 조합이 그것이고, 러너는
    # `from_pretrained` 뒤에 대입해서 그 검사를 우회하므로 대입 전에 본다).
    # 이 스크립트는 체인 회차를 띄우는 통로이고 체인은 exec 30 으로 돌리므로
    # 더 좁게 두는 쪽이 맞다. 의도된 비대칭이다 — 맞추려 하지 마라.
    if checkpoint.get("temporal_ensemble_coeff") is not None:
        problems.append(
            "체크포인트에 temporal_ensemble_coeff 가 박혀 있다 -- "
            "n_action_steps>1 과 같이 못 쓴다"
        )

    exec_steps = model.get("n_action_steps") or chunk
    if int(exec_steps) > chunk:
        problems.append(f"n_action_steps {exec_steps} > chunk {chunk}")

    # 2라운드(env 토큰) 체크포인트: `observation.environment_state`
    # (`FeatureType.ENV`, shape [K]) 를 **추가로** 선언한다. state 는 그대로
    # 16+K 다 -- 같은 원핫을 ACT 인코더의 별도 토큰으로도 받는다
    # (`modeling_act.py:344-346, 465-466`). 거부하지 않는다; 폭만 본다.
    env_feature = (checkpoint.get("input_features") or {}).get(
        "observation.environment_state"
    )
    env_k = None
    if env_feature is not None:
        env_type = env_feature.get("type")
        if env_type != "ENV":
            problems.append(
                f"체크포인트의 observation.environment_state 가 type={env_type!r} 다 "
                "-- ACT 는 ENV 타입일 때만 토큰화하고, 다른 타입이면 정규화기가 "
                "건드린다"
            )
        try:
            env_k = int(env_feature["shape"][0])
        except (KeyError, IndexError, TypeError, ValueError):
            problems.append(
                "체크포인트의 observation.environment_state 에서 shape 를 못 읽었다"
            )

    onehot_k = model.get("onehot_k")
    if onehot_k:
        if state != 16 + int(onehot_k):
            problems.append(
                f"체크포인트 state={state}D 인데 원핫 K={onehot_k} 는 "
                f"16+{onehot_k}={16 + int(onehot_k)}D 를 요구한다"
            )
        if env_k is not None and env_k != int(onehot_k):
            problems.append(
                f"체크포인트 env 토큰 {env_k}D 인데 원핫 K={onehot_k} 다 -- "
                "env 토큰은 state 꼬리와 **같은** 원핫이라 폭이 같아야 한다"
            )
    elif env_k is not None:
        # state 가 16 이어도 여기서 걸린다: 원핫을 안 주면 그 키를 채우는 것이
        # 없고, ACT 는 첫 프레임에 `batch["observation.environment_state"]` 를
        # 읽는다 -- robot.connect() 뒤다.
        problems.append(
            f"체크포인트가 env 토큰({env_k}D, 2라운드)을 선언했는데 YAML 의 "
            "chain.model.onehot_k 가 비어 있다 -- 그 키를 채우는 것이 없어 첫 "
            "프레임에 죽는다. onehot_k 를 학습 단계 수로 채워라"
        )
    elif state not in (14, 16):
        problems.append(
            f"체크포인트 state={state}D -- 원핫 모델이면 YAML 의 "
            "chain.model.onehot_k 를 채워라"
        )
    # state 의 원핫 꼬리와도 대조한다 (위 K 검사와 별개 -- 체크포인트가 스스로
    # 선언한 두 값이 어긋나는 경우를 따로 집어낸다).
    if env_k is not None and env_k != state - 16:
        problems.append(
            f"체크포인트 env 토큰 {env_k}D 인데 state={state}D 의 원핫 꼬리는 "
            f"{state - 16}D 다 -- 학습된 모델에선 같은 벡터다"
        )

    expected_progress = {16: False, 17: True}.get(action)
    if expected_progress is None:
        problems.append(
            f"체크포인트 action={action}D -- 체인은 16(진행도 없음)과 "
            "17(progress)만 안다. 로봇 action 폭은 16 이고, 그보다 한 칸 넓은 "
            "것만 진행도로 해석한다"
        )
    # `has_progress` 는 **명시 필수**이고 폭과 **둘 다** 맞아야 한다. 러너의
    # preflight 가 같은 것을 거부하지만, 여기서 먼저 막는 이유는 거부 지점이
    # 로봇 연결·safetensors 다운로드보다 앞이기 때문이다.
    declared = model.get("has_progress")
    if declared is None:
        problems.append(
            f"{yaml_path} 의 chain.model.has_progress 가 null(또는 없음)이다 — "
            f"명시해야 한다. 이 체크포인트는 action={action}D 이므로 "
            f"{'true' if expected_progress else 'false'} 다. 「폭에서 읽는다」로 "
            "두면 녹화 데이터셋의 action 칸 수(=정규화기 폭)를 정하는 사실이 "
            "설정에 남지 않는다"
        )
    elif expected_progress is not None and bool(declared) != expected_progress:
        problems.append(
            f"YAML 의 chain.model.has_progress={declared} 인데 체크포인트 "
            f"action={action}D 는 {expected_progress} 다"
        )

    fps = dataset.get("fps")
    if not fps:
        problems.append(f"{yaml_path} 에 dataset.fps 가 없다")

    params = None
    if fps:
        try:
            params = load_chain_params(params_path, expected_fps=int(fps))
        except ChainParamsError as error:
            problems.append(str(error))

    if problems:
        for problem in problems:
            print(f"!! {problem}", file=sys.stderr)
        return 3

    print(
        f"   체크포인트 state {state}D · action {action}D "
        f"({'progress 있음' if action == 17 else '진행도 없음'}) · "
        f"chunk {chunk} · exec {exec_steps} · fps {fps} · "
        f"{f'env 토큰 {env_k}D (2라운드)' if env_k is not None else 'env 토큰 없음 (1라운드)'}"
        " 확인",
        file=sys.stderr,
    )
    assert params is not None
    print(
        f"   단계 파라미터 OK: {len(params.stages)}단계 · fps {params.fps} · "
        f"sim {str(params.source.get('sim_commit', '?'))[:12]} · "
        # 10/07: 이동 단계 목록. 비어 있으면 옛 파일 -- 이동 단계가 베이스 없이 완료될 수 있다(1756).
        f"move={[n for n, s in sorted(params.stages.items()) if s.kind == 'move']}",
        file=sys.stderr,
    )

    # 셸이 다시 읽는 줄. 원핫 모델이면 `set_stage` 가 이 체크아웃에 있는지 보게 한다.
    if onehot_k:
        print(f"ONEHOT_K={int(onehot_k)}")
    # 셸은 `ONEHOT_K=` 만 grep 한다 (eval_chain.sh:136). 이 줄은 회차 로그에
    # 어느 라운드의 체크포인트였는지 남기기 위한 것이다.
    if env_k is not None:
        print(f"ENV_K={env_k}")
    print(f"ACTION_DIM={action}")
    print(f"STATE_DIM={state}")
    print(f"FPS={int(fps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
