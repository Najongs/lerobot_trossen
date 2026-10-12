"""SmolVLA ActionQuery — VLA-Adapter(arXiv 2509.09372) 의 학습 쿼리 토큰을 SmolVLA prefix 에 붙인다 (sim §94.50, J38).

무엇을: VLM 입력(prefix = 이미지 · 언어 · state) **맨 뒤**에 학습 토큰 N 개를 새 attention 블록으로 붙인다.
  - att 마스크는 ``[1, 0, …, 0]`` -- ``make_att_2d_masks`` 규칙상 앞 토큰(이미지·언어·state)은 쿼리를 **못 보고**, 쿼리는 앞 전부와
    서로를 본다. 그래서 동결 VLM 의 이미지·언어 처리는 **그대로**이고, 쿼리만 「행동에 필요한 것」 을 모으는 자리가 된다.
  - expert 는 층마다 prefix 전체의 K/V 를 cross-attention 으로 읽으므로(``smolvlm_with_expert``) 쿼리 출력이 자동으로 조건에 들어간다.
    VLA-Adapter 의 ①(층별 raw 특징 주입)은 SmolVLA 에 이미 있고, 이것이 없는 ②다. 논문의 주입 게이트 tanh(g) 는 넣지 않는다
    (SmolVLA 의 cross-attn 이 이미 학습되는 경로라 첫 버전은 최소 변경).
  - 초기값: 어휘 임베딩 N 행(고정 시드) × √H -- ``embed_prefix`` 가 언어 임베딩에 곱하는 배율과 같아 동결 VLM 이 보는 입력 분포 안에서 시작.

활성 신호는 **가중치 파일 자체**다: ``model.action_queries`` 키가 있으면 ``install_load_hook`` 이 로드 직전에 붙인다. lerobot 의
``from_pretrained`` 는 ``strict=False`` 라 이 훅 없이 읽으면 쿼리가 **조용히 버려진** 채 돈다 -- 그래서 로봇 플러그인이 import 때 훅을 건다.
이 파일은 torch·lerobot 외에는 import 하지 않는다(로봇 SDK 무관). **lerobot_trossen 의 같은 이름 파일과 바이트 동일해야 한다.**

**훅은 안전망이 아니다 -- ``verify`` 가 안전망이다** (codex 10/12): spawn 워커(``record_ensemble.py``)처럼 플러그인을 안 거치는 로드
경로가 있다. 그래서 쿼리가 실릴 수 있는 로드 지점마다 로드 **뒤에** ``verify(policy, 체크포인트)`` 를 불러, 파일에 쿼리 키가 있는데
모델에 쿼리가 없거나 값이 다르면 예외로 멈춘다.
"""
import math
import types

import torch
from torch import nn

KEY = "model.action_queries"
_SEED = 20261012


def attach(flow, n: int) -> bool:
    """``flow`` (= ``SmolVLAPolicy.model``, ``VLAFlowMatching``) 에 쿼리 ``n`` 개를 붙인다. 이미 있으면 모양만 확인. 새로 붙였으면 True."""
    if n <= 0:
        raise ValueError(f"action queries n={n} -- 양수여야 한다")
    if getattr(flow, "prefix_length", -1) and getattr(flow, "prefix_length", -1) > 0:
        raise ValueError("prefix_length > 0 (prefix 패딩) 에서는 쿼리 위치가 패딩 뒤로 밀린다 -- 지원하지 않는다")
    old = getattr(flow, "action_queries", None)
    if old is not None:
        if old.shape[0] != n:
            raise ValueError(f"이미 붙은 쿼리 {tuple(old.shape)} 와 n={n} 이 다르다")
        return False
    emb = flow.vlm_with_expert.get_vlm_model().text_model.get_input_embeddings().weight  # (V, H)
    h = emb.shape[1]
    g = torch.Generator().manual_seed(_SEED)
    idx = torch.randint(0, emb.shape[0], (n,), generator=g)
    init = emb.detach().float().cpu()[idx] * math.sqrt(h)
    flow.action_queries = nn.Parameter(init.to(emb.device))  # fp32 -- 학습 파라미터

    # 원본은 **클래스 함수**로 부른다 -- 인스턴스의 bound method 를 클로저로 잡으면 deepcopy 한 사본이 원본 인스턴스를 계속 쓴다
    def embed_prefix(self, *args, **kwargs):
        embs, pad_masks, att_masks = type(self).embed_prefix(self, *args, **kwargs)
        b = embs.shape[0]
        q = self.action_queries.to(device=embs.device, dtype=embs.dtype)[None].expand(b, -1, -1)
        qa = torch.zeros(b, q.shape[1], dtype=att_masks.dtype, device=att_masks.device)
        qa[:, 0] = True
        return (torch.cat([embs, q], dim=1),
                torch.cat([pad_masks, torch.ones(b, q.shape[1], dtype=pad_masks.dtype, device=pad_masks.device)], dim=1),
                torch.cat([att_masks, qa], dim=1))

    flow.embed_prefix = types.MethodType(embed_prefix, flow)
    return True


def install_load_hook(policy_base=None) -> None:
    """``PreTrainedPolicy._load_as_safetensor`` 를 감싸 파일에 ``KEY`` 가 있으면 로드 전에 ``attach`` 한다. 두 번 불러도 한 번만 건다."""
    if policy_base is None:
        from lerobot.policies.pretrained import PreTrainedPolicy as policy_base
    if getattr(policy_base, "_action_query_hooked", False):
        return
    orig = policy_base._load_as_safetensor.__func__

    def _load_as_safetensor(cls, model, model_file, *args, **kwargs):
        from safetensors import safe_open
        with safe_open(str(model_file), "pt") as f:
            if KEY in f.keys():
                n = int(f.get_slice(KEY).get_shape()[0])
                flow = getattr(model, "model", None)
                if flow is None or not hasattr(flow, "vlm_with_expert"):
                    raise RuntimeError(f"{model_file}: {KEY} 가 있는데 SmolVLA 모델이 아니다 ({type(model).__name__})")
                attach(flow, n)
                print(f"[action_query] {KEY} ({n}) 를 붙여 싣는다 -- {model_file}")
                out = orig(cls, model, model_file, *args, **kwargs)
                verify(out, model_file)  # 실린 값이 파일과 같은가 -- 로드 뒤 바로
                return out
        return orig(cls, model, model_file, *args, **kwargs)

    policy_base._load_as_safetensor = classmethod(_load_as_safetensor)
    policy_base._action_query_hooked = True


def _model_file(checkpoint) -> str:
    from pathlib import Path
    p = Path(str(checkpoint))
    if p.is_file():
        return str(p)
    if (p / "model.safetensors").is_file():
        return str(p / "model.safetensors")
    from huggingface_hub import hf_hub_download  # 허브 id -- from_pretrained 가 이미 받아 둔 캐시를 가리킨다
    return hf_hub_download(repo_id=str(checkpoint), filename="model.safetensors")


def verify(policy, checkpoint) -> int:
    """로드 **뒤** 확인: 파일의 쿼리 유무·값 == 모델. 어긋나면 RuntimeError. 쿼리 수(없으면 0)를 돌려준다."""
    from safetensors import safe_open
    f = _model_file(checkpoint)
    with safe_open(f, "pt") as fh:
        want = fh.get_tensor(KEY) if KEY in fh.keys() else None
    flow = getattr(policy, "model", None)
    got = getattr(flow, "action_queries", None) if flow is not None else None
    if want is None and got is None:
        return 0
    if want is None:
        raise RuntimeError(f"[action_query] 모델에 쿼리가 있는데 {f} 에는 {KEY} 가 없다")
    if got is None:
        raise RuntimeError(f"[action_query] {f} 에 {KEY} {tuple(want.shape)} 가 있는데 모델에 안 실렸다 -- "
                           f"load hook 이 이 프로세스에 없다. 쿼리 없이 돌리지 마라")
    if tuple(got.shape) != tuple(want.shape) or not torch.equal(got.detach().float().cpu(), want.float()):
        raise RuntimeError(f"[action_query] 실린 쿼리가 {f} 의 값과 다르다")
    return int(want.shape[0])
