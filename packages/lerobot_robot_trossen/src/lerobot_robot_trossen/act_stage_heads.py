"""ACT 단계별 출력 헤드 + 단계별 디코더 쿼리 오프셋 — 단계가 **파라미터를 고른다** (sim §94.63, 10/15).

왜: 원핫을 「더 잘 주는」 변형(ENV 토큰·진행도·드롭아웃·언어)은 전부 실패했다 -- 학습 데이터 안에서 영상·state 가 단계를
완전히 정해 원핫이 잉여이기 때문이다(``train_multi.py`` ``boundary_hold`` 절, §89·§94.6). 잉여 입력은 무시되지만,
출력층이 단계마다 따로면 단계 k 의 헤드는 단계 k+1 의 라벨을 **아예 보지 않는다** -- 무시할 수 없는 조건이다.

무엇을: ACT(``ACTPolicy.model``)에 두 가지를 붙인다.
  - ``action_head`` 를 ``StageLinear`` 로 바꾼다 -- 가중치 ``(K, out, D)``·편향 ``(K, out)``. 원래 ``nn.Linear`` 를 K 벌 **복사**해 시작.
  - ``stage_query_offset`` ``(K, chunk, D)`` -- 디코더 위치 임베딩(= 행동 쿼리)에 단계별로 더한다. **0 으로** 시작.
  그래서 붙인 직후 출력은 원 모델과 같다(``check_identity`` 로 수치 확인).
단계 고르기: ``ACT.forward(batch)`` 의 ``observation.state`` **마지막 K 칸**(원핫, 정규화 통계 평균 0·표준편차 1 이라 0/1 그대로)의
argmax. 원핫이 없는 행(전부 0 -- ``stage_cond_diag`` 의 hot_zero 교란)은 K 개 헤드의 **평균**으로 간다(조건 없음 대용, 학습에는 없는 입력).

활성 신호는 **가중치 파일 자체**다: ``model.action_head.weight`` 가 3차원이면 ``install_load_hook`` 이 로드 직전에 붙인다.
훅이 없는 로더로 읽으면 ``action_head`` 모양이 맞지 않아 **크게 실패한다**(조용히 버려지지 않는다 -- lerobot ``strict=False`` 도
모양 불일치는 예외). 그래도 쿼리 오프셋이 함께 실렸는지는 ``verify`` 로 로드 뒤에 확인한다.
이 파일은 torch·lerobot 외에는 import 하지 않는다(로봇 SDK 무관). 로봇 쪽에 쓸 때는 lerobot_trossen 에 **바이트 동일한 사본**을 둔다.
"""
import torch
from torch import nn

HEAD_KEY = "model.action_head.weight"
QUERY_KEY = "model.stage_query_offset"
STATE_KEY = "observation.state"


class StageLinear(nn.Module):
    """단계마다 다른 ``Linear(D → out)``. ``self.stage_w`` (B, K) 로 섞는다(원핫이면 그 단계 하나)."""

    def __init__(self, k: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(k, d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(k, d_out))
        self.stage_w = None

    def forward(self, x):  # x (B, S, D)
        w = self.stage_w
        if w is None or w.shape[0] != x.shape[0]:
            raise RuntimeError("[stage_heads] 단계 가중치가 없다 -- ACT.forward 앞의 단계 훅이 돌지 않았다")
        W = torch.einsum("bk,kod->bod", w, self.weight)
        b = w @ self.bias
        return torch.einsum("bsd,bod->bso", x, W) + b[:, None, :]


def _is_stage(head) -> bool:
    """모듈 사본이 둘 이상 실려도(spec 로더마다 클래스가 따로 생긴다) 같은 판정이 나오게 이름·모양으로 본다."""
    return type(head).__name__ == "StageLinear" and getattr(head, "weight", None) is not None and head.weight.ndim == 3


def stage_weights(state: torch.Tensor, k: int) -> torch.Tensor:
    """state (B, 16+K) 의 마지막 K 칸 → (B, K) 라우팅 가중치. 원핫이면 그 칸 1, 전부 0 이면 1/K 균등."""
    hot = state[:, -k:].float()
    top, idx = hot.max(dim=-1)
    w = torch.nn.functional.one_hot(idx, k).to(hot.dtype)
    none = top < 0.5
    if none.any():
        w[none] = 1.0 / k
    return w


def _act_pre_hook(module, args, kwargs):
    batch = args[0] if args else kwargs.get("batch")
    k = module.stage_query_offset.shape[0]
    w = stage_weights(batch[STATE_KEY], k).to(module.stage_query_offset.dtype)
    module.action_head.stage_w = w
    # (B, K) x (K, S, D) -> (S, B, D): 디코더 위치 임베딩 (S, 1, D) 에 더할 단계별 쿼리
    module.decoder._stage_query_add = torch.einsum("bk,ksd->sbd", w, module.stage_query_offset)
    return None


def _decoder_pre_hook(module, args, kwargs):
    add = getattr(module, "_stage_query_add", None)
    if add is None:
        raise RuntimeError("[stage_heads] 디코더 쿼리 오프셋이 없다 -- ACT.forward 앞의 단계 훅이 돌지 않았다")
    pe = kwargs.get("decoder_pos_embed")
    if pe is None:
        raise RuntimeError("[stage_heads] ACTDecoder 가 decoder_pos_embed 를 키워드로 받지 않았다 -- lerobot 버전 확인")
    kwargs = dict(kwargs)
    kwargs["decoder_pos_embed"] = pe + add.to(pe.dtype)
    return args, kwargs


def _hooked(act) -> bool:
    return bool(getattr(act, "_stage_heads_hooked", False))


def _hook(act) -> None:
    if _hooked(act):
        return
    act.register_forward_pre_hook(_act_pre_hook, with_kwargs=True)
    act.decoder.register_forward_pre_hook(_decoder_pre_hook, with_kwargs=True)
    act._stage_heads_hooked = True


def attach(act, k: int) -> bool:
    """``act`` (= ``ACTPolicy.model``) 에 단계 헤드 K 개를 붙인다. 이미 있으면 모양만 확인. 새로 붙였으면 True.

    새로 붙일 때는 원래 ``action_head`` 를 K 벌 복사하고 쿼리 오프셋은 0 -- 출력이 원 모델과 같다."""
    if k < 2:
        raise ValueError(f"stage heads K={k} -- 2 이상이어야 한다")
    head = act.action_head
    if _is_stage(head):
        if head.weight.shape[0] != k or act.stage_query_offset.shape[0] != k:
            raise ValueError(f"이미 붙은 단계 헤드 K={head.weight.shape[0]} 와 K={k} 가 다르다")
        _hook(act)
        return False
    if not isinstance(head, nn.Linear):
        raise ValueError(f"action_head 가 nn.Linear 가 아니다 ({type(head).__name__}) -- ACT 가 아닌가")
    state_dim = act.config.robot_state_feature.shape[0] if act.config.robot_state_feature else 0
    if state_dim < 14 + k:
        raise ValueError(f"state {state_dim} 칸에 원핫 K={k} 를 둘 자리가 없다 (16+K 기대)")
    new = StageLinear(k, head.in_features, head.out_features).to(head.weight.device, head.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(head.weight.detach()[None].expand(k, -1, -1))
        new.bias.copy_(head.bias.detach()[None].expand(k, -1))
    act.action_head = new
    s, d = act.decoder_pos_embed.weight.shape
    act.stage_query_offset = nn.Parameter(torch.zeros(k, s, d, device=head.weight.device, dtype=head.weight.dtype))
    _hook(act)
    return True


def extra_params(act) -> int:
    """붙여서 늘어난 파라미터 수 (원 ``Linear`` 대비)."""
    h = act.action_head
    k, o, d = h.weight.shape
    return (k - 1) * (o * d + o) + act.stage_query_offset.numel()


def install_load_hook(policy_base=None) -> None:
    """``PreTrainedPolicy._load_as_safetensor`` 를 감싸 파일의 ``HEAD_KEY`` 가 3차원이면 로드 전에 ``attach`` 한다. 두 번 불러도 한 번만."""
    if policy_base is None:
        from lerobot.policies.pretrained import PreTrainedPolicy as policy_base
    if getattr(policy_base, "_stage_heads_hooked", False):
        return
    orig = policy_base._load_as_safetensor.__func__

    def _load_as_safetensor(cls, model, model_file, *args, **kwargs):
        from safetensors import safe_open
        with safe_open(str(model_file), "pt") as f:
            keys = set(f.keys())
            shape = f.get_slice(HEAD_KEY).get_shape() if HEAD_KEY in keys else None
        if shape is not None and len(shape) == 3:
            act = getattr(model, "model", None)
            if act is None or not hasattr(act, "decoder_pos_embed"):
                raise RuntimeError(f"{model_file}: 단계 헤드 {HEAD_KEY} {shape} 가 있는데 ACT 가 아니다 ({type(model).__name__})")
            if QUERY_KEY not in keys:
                raise RuntimeError(f"{model_file}: 단계 헤드는 있는데 {QUERY_KEY} 가 없다")
            attach(act, int(shape[0]))
            model._stage_heads_file = str(model_file)
            print(f"[stage_heads] 단계 헤드 K={shape[0]} 를 붙여 싣는다 -- {model_file}")
            out = orig(cls, model, model_file, *args, **kwargs)
            verify(out, model_file)
            return out
        return orig(cls, model, model_file, *args, **kwargs)

    policy_base._load_as_safetensor = classmethod(_load_as_safetensor)
    policy_base._stage_heads_hooked = True


def _model_file(checkpoint) -> str:
    from pathlib import Path
    p = Path(str(checkpoint))
    if p.is_file():
        return str(p)
    if (p / "model.safetensors").is_file():
        return str(p / "model.safetensors")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=str(checkpoint), filename="model.safetensors")


def verify(policy, checkpoint=None) -> int:
    """로드 **뒤** 확인: 파일의 단계 헤드 유무·값 == 모델, 훅이 걸려 있나. 어긋나면 RuntimeError. K(없으면 0)를 돌려준다."""
    from safetensors import safe_open
    f = getattr(policy, "_stage_heads_file", None) or _model_file(checkpoint)
    with safe_open(f, "pt") as fh:
        keys = set(fh.keys())
        w = fh.get_tensor(HEAD_KEY) if HEAD_KEY in keys else None
        q = fh.get_tensor(QUERY_KEY) if QUERY_KEY in keys else None
    act = getattr(policy, "model", None)
    staged = _is_stage(getattr(act, "action_head", None))
    if w is None or w.ndim != 3:
        if staged:
            raise RuntimeError(f"[stage_heads] 모델에 단계 헤드가 있는데 {f} 에는 없다")
        return 0
    if not staged:
        raise RuntimeError(f"[stage_heads] {f} 에 단계 헤드 {tuple(w.shape)} 가 있는데 모델에 안 실렸다 -- load hook 이 없다")
    for name, want, got in (("head", w, act.action_head.weight), ("query", q, act.stage_query_offset)):
        if want is None or tuple(got.shape) != tuple(want.shape) or not torch.equal(got.detach().float().cpu(), want.float()):
            raise RuntimeError(f"[stage_heads] 실린 {name} 가 {f} 의 값과 다르다")
    if not _hooked(act):
        raise RuntimeError("[stage_heads] 단계 헤드는 있는데 라우팅 훅이 없다")
    return int(w.shape[0])
