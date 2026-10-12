"""오프라인 SmolVLA 도구용 ActionQuery(J38) 가드 — import 하면 load hook 이 걸린다, 로드 뒤 ``aq.verify(policy, path)``.

이 도구들은 플러그인을 거치지 않고 ``SmolVLAPolicy.from_pretrained`` 를 직접 부른다. lerobot 은 ``strict=False`` 라 훅이
없으면 J38 체크포인트의 쿼리가 버려진 채 「정상 결과」 가 나온다(codex 10/12). 로봇 SDK 를 끌어오지 않도록 패키지가 아니라
파일 경로로 연다. 쿼리 키가 없는 체크포인트에는 출력 무영향.
"""
import importlib.util
from pathlib import Path

_P = (Path(__file__).resolve().parents[2]
      / "packages/lerobot_robot_trossen/src/lerobot_robot_trossen/smolvla_action_query.py")
_spec = importlib.util.spec_from_file_location("smolvla_action_query", _P)
aq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aq)
aq.install_load_hook()
