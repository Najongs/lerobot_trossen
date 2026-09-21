# 2026-09-21 SmolVLA 추론 실험 원본

- 정리된 기록: [../../smolvla_inference_experiments.md](../../smolvla_inference_experiments.md) (실험 E1~E22), [../../smolvla_run_index.md](../../smolvla_run_index.md) (런 R1~R40 표)
- `*.log`: 로봇 런 40개의 원본 로그. 런 번호와 파일의 대응은 런 표 문서 맨 아래에 있다.
- `bench/`: 오프라인 측정에 쓴 일회용 스크립트. 모델·데이터셋 경로가 이 로봇 PC 기준으로 박혀 있고, 일부는 `/tmp/smolbench/`를 가리킨다. 다시 돌리려면 경로부터 고쳐야 한다.

| 스크립트 | 실험 |
|---|---|
| `nsteps.py` | E8 디노이징 스텝 수별 추론 시간 (E2의 10스텝 수치도 여기서 다시 나온다. E2에 쓴 원래 스크립트는 남아 있지 않다) |
| `compile_test.py` | E3 `torch.compile` 측정. config를 덮어쓰지 못해 **무효**였던 버전이다 |
| `repro.py`, `repro2.py`, `repro3.py`, `repro4.py` | E6 합성 부하 재현 시도 |
| `bf16diff.py` | E9 fp32 대 bf16 액션 차이 |
| `kens.py` | E11 노이즈 축소, K-샘플 평균 |
| `gil_base.py` | E14 base 드라이버의 GIL 점유. **base 시리얼 포트를 연다**(읽기 전용) |
| `e2e.py` | E15 스레드 워커 대 프로세스 워커. **base 시리얼 포트를 연다**(읽기 전용). `e2e.py process 150`처럼 실행한다 |
| `knob_test.py`, `merge_test.py` | `noise_scale`·`samples`·`merge` 옵션의 CPU 로직 테스트 |
| `prefetch_test.py`, `commit_test.py`, `commit_ta_test.py` | E18~E20 CPU 시뮬레이션 |
| `acc_eval.py`, `acc_eval2.py` | E22 시연 대비 정확도. `acc_eval2.py <모델 경로> <데이터셋 경로>`는 돌리지 못한 후속 비교용이다 |
| `index_runs.py`, `make_run_doc.py` | 로그에서 런 표를 다시 만든다. `python3 bench/index_runs.py > index.md` 후 `python3 bench/make_run_doc.py <index.md가 있는 폴더>` |
