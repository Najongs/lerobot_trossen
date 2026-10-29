# 베이스 이동이 섞인 IL/VLA 논문 — 이동 관련 문제와 대응 (10/28 조사)

서브에이전트가 arXiv MCP 로 21편을 받아 전문 검색으로 정리한 것. 우리 문제: ① 직진 오도 ≈2% 과대 ② 캐스터 이력에 따라 같은 회전 명령이 −2~+6° ③ 루프 Hz 불일치 과회전 ④ 재생 시작 자세가 녹화와 달라 첫 회전이 틀어짐. 우리 해법: cam_high 로 닫는 회전(`docs/depth_cam_high_pc2_1027.md`).
「못 찾음」은 전문 키워드 0히트 확인. 볼트 반증 목록과는 대조 못 함(PC2 에 vaultgraph 없음).

## 핵심 5줄
1. **①②③ 은 문헌에 사실상 없다.** 오도 스케일을 측정·보정한 논문 1편(TidyBot++, holonomic·모캡), 캐스터 이력 회전 0편, Hz 불일치 과회전 0편.
2. **문헌의 처방은 「베이스를 정확히 보내라」가 아니라 「베이스를 action 에서 빼라 / 조작 직전에 비전으로 재정렬하라」로 수렴.** HoMeR·MoMa-Teleop·MoManipVLA 는 베이스를 정책 action 에서 제거(EE pose → IK/WBC/RL), Last-meter·Cracking MoMa·ANCHOR·SVM 은 조작 직전 비전 재정렬.
3. **베이스를 action 에 직접 넣은 논문은 전부 속도 명령인데, 전부 holonomic 이거나 회전 모드를 봉인했다.** π0.5(선속도 2D+각속도, 50 Hz)·Mobile ALOHA(16D)·PanoVLA(17D)·MobRT. MobRT 는 「모드 전환 지연을 피하려고」 translation 만. **nonholonomic 에서 속도 명령으로 정확히 돌린 선례는 없다.**
4. **시작 자세(④)는 문헌이 정면으로 다루고 수치가 크다.** N2M: 데이터 수집과 같은 자세 0.93/0.97 → 기하학적으로 도달만 가능한 자세 0.16/0.15. MoMa-Teleop: 베이스를 action 에 넣은 정책이 서랍 높이 변경에서 100→0. MoManipVLA: 실패의 72% 가 「타깃에 정렬되지 않음」.
5. **차동구동 자체가 측정된 페널티.** TidyBot++: 같은 50 시연·500 epoch 로 holonomic 9/10 vs 차동구동 4/10, 경로 2.03 m/27 s vs 4.03 m/65 s. 이유 ⓐ 평행주차식 측면 이동을 추가로 배워야 함 ⓑ 차동구동 기동이 카메라 시야를 좌우로 흔듦.

## 표 A — 베이스 action 표현 / 지연·주기 / 오도·외부 피드백 (— = 못 찾음, ★ = 우리와 직결)
| 논문 | action 표현 (왜) | 지연·주기 | 오도 / 외부 피드백 |
|---|---|---|---|
| 2401.02117 Mobile ALOHA | 선속도+각속도 → 팔 14D 와 concat 16D | ★**base delay d step 측정 → 청크 k 에서 팔은 앞 k−d, 베이스는 뒤 k−d 실행** | CAN 바퀴 속도. SLAM 없이 정책이 재생 오차를 교정한다고 명시 |
| 2412.10447 TidyBot++ | ★목표 pose (x,y,θ) position mode (holonomic 이라 가능; 「nonholonomic 은 velocity mode 밖에 안 된다」) | — | ★모캡으로 오도 정량화: 병진 <1 cm/m, 회전 <1°/360° |
| 2506.01185 HoMeR | ★베이스를 action 에서 제거, EE SE(3) → WBC(IK) | 10 Hz | WBC 에 base damping W=1.5 |
| 2502.13964 SVM | ★비전 서보잉이 전신 속도 명령(닫힌 루프), nonholonomic Stretch | — | ★손목 RGB-D 로 닫음. 「nonholonomic 베이스는 정밀 재위치를 제한」 |
| 2503.13446 MoManipVLA | world-frame 목표 pose. velocity 0히트 | 공간 step 0.05 | ★바퀴 오도 안 씀 — T265 + ORB-SLAM |
| 2608.02257 PanoVLA | 17D = 팔 14 + (v_x,v_y,ω). state 에 시작프레임 기준 적분 pose | 25 Hz, 32-step 청크, 매 주기 주행모드 재평가 | — . 360° 파노라마로 전역 재탐지 |
| 2512.11173 Last-meter | ★이산 프리미티브 병진 0.2 m·회전 12° = 「Spot 이 신뢰성 있게 실행할 최소 증분」 | — | 오도·depth 의도적 배제(RGB only), GT 는 AprilTag |
| 2608.01381 DreamTrajectory | ★속도를 **청크 구간에 적분**해 chunk-local frame EE 궤적으로 감독 | H=16, 후보 30개 월드모델 검증 | ★해석적 FK 는 실현 궤적을 xyz 0.241 m·27.0° 틀림, 학습형 GRU 0.028 m·6.2° |
| 2603.22760 SG-VLA | ΔX (x,y,yaw 변위) | 청크 8 중 앞 2 실행 | 보조 디코더로 전역 (x,y) 재구성 |
| 2606.12956 SERF | — | ★30-step 중 앞 26 → cubic 보간 → 20 명령 | ★「dead reckoning 은 심하게 드리프트」 → ORB-SLAM3 (ATE 2.42 cm·0.34°) |
| 2610.08220 VOMMI | ★body frame 속도 (v_x,v_y,ψ̇), 베이스 차원 그룹 분리 | ★15 Hz 단일 클럭 리샘플, 다중 호라이즌 토큰 | ★VO 는 「장거리·큰 회전」에 취약 → 1 Hz VGGT 앵커, ATE 15.65→5.97 cm |
| 2509.25032 AIRoA MoMa | ★상대 증분 (Δx,Δy,Δθ), 출처는 텔레옵 명령. 「**오도 정확도가 낮아 절대 베이스 상태·action 을 싣지 않았다**」 | 30 Hz 리샘플, 결측 `stale` 표시 | — |
| 2510.04592 MobRT | 베이스 velocity. ★「모드 전환 지연을 피하려고 translation 만」 | 시뮬이 actuation delay 를 못 담아 실데이터 co-train | — |
| 2409.15095 MoMa-Teleop | ★★사람도 정책도 베이스를 명령 안 함 — EE 6D 만, 베이스는 RL(N²M²) | 30/80 Hz | — |
| 2504.16054 π0.5 | ★선속도 2D+각속도, proprio 에 베이스 속도 포함(우리 16D 와 같은 구조), holonomic | 50 Hz + 청크 H=49, PD 추종 | — |
| 2610.19600 Cracking MoMa | 베이스 조정 스킬은 A* + PD | 20 Hz, horizon 50 | 매 시점 depth 로 점유격자 재구성(누적 추측항법 아님) |
| 2604.25323 ANCHOR v1 | SE(2) 자세 PSO 최적화 → 네비 스택 | — | LiDAR SLAM. ⚠ 사족보행 |
| 2606.00253 Per-Group · 2302.04334 Asking for Help · 2503.20020 Gemini Robotics(모바일 베이스 없음) | — | — | — |

## 표 B — 시작 자세 대응 / 이동 관련 실패 수치
| 논문 | 시작 자세 | 보고된 실패·수치 |
|---|---|---|
| Mobile ALOHA | — | ★★**개방루프 재생**: 180° 선회·6 s 궤적 20회 → 전부 왼쪽 ≈10 cm 치우치고 ≈20 cm 선상 퍼짐. 「베이스 속도 제어의 확률성」이 실패 주원인 |
| TidyBot++ | — | ★★차동구동 ablation 9/10 vs 4/10. 캐스터 조향 마찰로 backdrive 가 잘 안 됨 |
| HoMeR | ★벤치마크에 randomized robot poses 축. 장거리는 절대 keypose(「step-wise delta 보다 안정」) | DP(베이스+팔 dense)가 랜덤 자세에서 가장 많이 실패 |
| SVM | — | ★★개방루프 7/42·12/42 → 닫힌루프 30/42 |
| N2M | ★★viewpoint aug(300) + positional aug(Z 회전 + 1 m 병진) | ★★Oracle 0.93/0.97 vs Reachability 0.16/0.15. 「초기 베이스 자세에 극단적 민감」 |
| MoManipVLA | 베이스 proprio 로 waypoint 를 world frame 에 고정 | ★★실패 72.09% = Orient_to_place |
| Per-Group | 초기 조건 통제(한계로 명시) | ★★베이스 그룹 MSE 가 가장 늦게 수렴하는데, **베이스 오차가 더 낮은 모델이 실기에서 졌다**(3.75/4 vs 4.0/4, p=0.010) — 배포 순위와 맞은 오프라인 지표는 팔 그룹 오차뿐 |
| Last-meter | ★★시작 pose 격자(반경 5단·접근각 15°·heading −150~150° 30°) | ★허용오차 0.2 m/±6°. 실패 1위 = stop 을 못 내 지나침, 원인 「Spot 이 미세 조정을 못 해 데모 종단 상태가 변동」 |
| DreamTrajectory | — | 청크 초반 작은 편차가 접촉·다음 관측을 동시에 바꿔 누적 |
| SG-VLA | — | flow matching 연속 헤드가 이동 과제를 깎음(drawer 0.77→0.60) — 「이동에는 이산 action 토큰이 더 적합」 |
| VOMMI | ★상대 action 이 전역 pose 오차에 둔감하다는 수식 논증 | Base MAE 0.044→0.036 m/s |
| AIRoA MoMa | ★초기 위치 랜덤화 | 실패 ≈6.6%, misalignment 가 베이스인지 미특정 |
| MobRT | 🔴베이스 시작 자세 고정(world 원점) | 베이스 실패 지목 없음 |
| MoMa-Teleop | ★★시연 간 **베이스 배치 전략 일관성**이 학습 성패를 가름 — 일관되지 않게 모으니 정책이 동작을 섞어 실패, 일관되게 하자 성공 | ★★베이스를 action 에 넣은 정책: Drawer New Height **0**, Obstacle **0** vs 베이스 위임 80·90 |
| Cracking MoMa | — | ★★실패 25.9% = navigation/base positioning. 「arrived-but-inoperable」 handoff 문제. marginal 이면 **heading 고정하고 평면 이동만 ≤0.25 m** |
| ANCHOR v1 | — | Align 추가로 grasp 69.2→80.0, 정렬 제거 −11.7 pp |

## 우리에게 바로 쓸 아이디어 (이미 만든 것 제외)
1. **방향 전환 지연을 「청크 인덱스 오프셋 d」로** — Mobile ALOHA §6: 팔은 앞 k−d, 베이스는 뒤 k−d. 3.6° ÷ 15°/s ≈ 0.24 s × 루프 Hz = d. `exec 30` 청크 안에서 바로 시험 가능. [추정] d 가 방향·캐스터 자세에 따라 달라 조건부일 수 있음.
2. **속도 재생 대신 「청크 구간 변위」로 감독·실행** — DreamTrajectory §3.4 + SERF App.F + VOMMI §V-C. 속도×dt 는 Hz 에 비례하지만 「이 청크 동안 몇 도」는 아니다. 기존 16D 베이스 2칸을 후처리로 변환 가능.
3. **스칼라 비 대신 명령 이력 조건부 「명령→실현」 매핑 학습** — DreamTrajectory: FK 27.0° vs GRU 6.2°. `base_calib_pc2.json` turn 칸이 null 인 것이 비(ratio) 모델이 틀렸다는 신호. `caster_test.py` 산출물이 학습 데이터.
4. **추론 시점 규칙 기반 정렬 오버라이드(학습 불변)** — Last-meter §III-D(마스크 IoU·질량중심 조건이면 stop 덮어씀) + Cracking MoMa D.3(heading 고정, 평면 이동만) + ANCHOR(−11.7 pp). 우리 `marker_yaw.py`/`base_turn.py` 가 판정·실행기.
5. **정책을 강건하게 만들기 전에 「시작 자세별 성공률 지도」** — N2M Table 1 설계(격자 × 셀당 5 trial). 메모리 목표 「흔들린 시작에서 70%」가 달성 가능한지, 시작 자세를 고르는 쪽이 싼지 먼저 답한다.

### ⚠ 반대 신호
- **출발 다양화가 역효과일 수 있다** — MoMa-Teleop §V-A(일관된 배치가 성패), MobRT(시작 고정) vs AIRoA·Last-meter(다양화). 문헌이 갈린다. 우리 t01 추가 녹화안(흔든 출발 30·회전 중간 출발 9)은 분포를 복잡하게 하는 쪽. **실기로 판정할 사안.**
- **베이스 오차를 줄여도 실기 성공률이 담보되지 않는다** — Per-Group §IV-D. 「베이스 보정이 개선됐다」는 실기 성공률로 역검증.

## 못 찾은 것
캐스터/방향 전환 이력 의존 회전 0편 · Hz 불일치 과회전 0편 · 오도 스케일 보정 1편(TidyBot++) · cm/deg 베이스 오차 수치는 6편뿐 · Gemini Robotics 는 모바일 베이스 없음 · Asking for Help 는 실패 모드에 베이스 없음(음성 결과).
표 수치 중 MoMa-Teleop Table III·SVM Table I·N2M Table 1 은 원문 재확인됨.
