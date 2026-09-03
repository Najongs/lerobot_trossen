#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lerobot>=0.5,<0.6",
#     "numpy",
#     "pandas",
# ]
# ///
"""LeRobot 데이터셋 벡터 feature 차원 슬라이스 — 일부 채널 제거 재파생.

[한국어]
LeRobot 데이터셋에서 한 벡터 feature(예: ``observation.state``)의 **일부 차원만
잘라내** 새 데이터셋으로 재파생한다. lerobot CLI(``lerobot-edit-dataset``)는
feature 전체(키 단위) 삭제만 지원하고, 한 feature 내부의 차원 단위 슬라이스는
못 하므로 ``dataset_tools`` Python API를 쓴다.

동기(motivating case): base 속도를 ``observation.state``(16-dim)에서 빼 14-dim
(팔만)으로 만들면 모방학습 copycat 자기강화를 피할 수 있다. base 속도는 캐논
(원본 Mobile ALOHA / Pi0)에서 ``action`` 전용이고 관측 state엔 넣지 않는다.

**원본 데이터셋은 절대 수정하지 않는다** — ``add_features`` / ``remove_feature``
는 항상 새 ``repo_id`` 사본을 만들고, 원본은 읽기 전용으로만 읽는다.

요구: lerobot >= 0.5.0 (``recompute_stats`` 는 0.5.0 이상에만 있음).
  -> **이 repo 의 env 로는 안 돈다** (Python 3.11 핀 -> lerobot 0.4.4).
     위 inline metadata 를 uv 가 읽어 전용 임시 env 를 만들도록 ``uv run --script``
     로 실행한다. ``uv run`` (--script 없이) 은 이 repo env 를 쓰므로 실패한다.
포맷: 입출력 모두 dataset codebase v3.0. lerobot 0.4.x 도 v3.0 이라 산출물을
0.4.x 학습 env 에서 그대로 학습할 수 있다(편집=0.5.x / 학습=0.4.x, 변환 불필요).

[English]
Re-derive a new LeRobot dataset with some dimensions removed from a vector
feature (e.g. drop the base-velocity channels from ``observation.state``). The
lerobot CLI can only remove whole features (key-level), not slice dimensions
within one, so this uses the ``dataset_tools`` Python API. The original dataset
is never modified (new ``repo_id`` copies are written). Requires lerobot >= 0.5
for ``recompute_stats``; output stays codebase v3.0 (trainable under 0.4.1).

절차 / Procedure (이름 충돌·remove 선행 가드 때문에 이 순서를 강제):
  (1) 슬라이스한 feature 를 임시 키로 추가        -> 새 사본 _tmp
  (2) 원본 feature 제거                            -> 새 사본 (출력)
  (3) 수동 rename: parquet 컬럼 + info.json 키     (임시 키 -> 원래 키)
  (4) recompute_stats  (>= 0.5.0)
  (5) (선택 --push) Hub 업로드

예시 / Examples:
  # base 제거: observation.state 앞 14-dim 만 유지(14, 15 드롭)
  uv run --script scripts/slice_feature_dims.py --repo-id ORG/dataset \
      --keep-first 14 --out-repo-id ORG/dataset_nobasestate
  # 인덱스로 명시
  uv run --script scripts/slice_feature_dims.py --repo-id ORG/dataset \
      --drop-indices 14,15 --out-repo-id ORG/dataset_nobasestate
  # 로컬 검증 후, 이미 만든 산출물만 업로드
  uv run --script scripts/slice_feature_dims.py --repo-id ORG/dataset \
      --keep-first 14 --out-repo-id ORG/dataset_nobasestate --push-only
"""

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import add_features, remove_feature, recompute_stats


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "벡터 feature 의 일부 차원을 잘라 새 LeRobot 데이터셋으로 재파생한다 "
            "(원본 불변). lerobot>=0.5 필요. / Slice dimensions out of a vector "
            "feature into a NEW dataset; requires lerobot>=0.5."
        )
    )
    parser.add_argument(
        "--repo-id", required=True, help="원본 데이터셋 repo_id / source repo_id"
    )
    parser.add_argument(
        "--feature",
        default="observation.state",
        help="슬라이스할 벡터 feature (기본: observation.state)",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--keep-first",
        type=int,
        metavar="N",
        help="앞 N 개 차원만 유지(나머지 드롭) / keep only the first N dims",
    )
    selection.add_argument(
        "--drop-indices",
        metavar="i,j,...",
        help="드롭할 차원 인덱스(쉼표 구분), 예: '14,15' / dim indices to drop",
    )
    parser.add_argument(
        "--out-repo-id",
        help=(
            "출력 repo_id (기본: {repo_id}_sliced). "
            "--push-only 일 때도 같은 값을 줘야 로컬 산출물을 찾는다"
        ),
    )
    parser.add_argument("--tmp-repo-id", help="중간 repo_id (기본: {repo_id}_tmp)")
    parser.add_argument(
        "--push",
        action="store_true",
        help="재파생 후 Hub 업로드 (기본 off — 먼저 로컬 검증 권장). hf auth login 필요",
    )
    parser.add_argument(
        "--push-only",
        action="store_true",
        help=(
            "재파생 건너뛰고 이미 만든 로컬 산출물만 업로드 "
            "(--out-repo-id 를 함께 줄 것. 이 모드에선 --push 는 무시된다)"
        ),
    )
    parser.add_argument("--private", action="store_true", help="private repo 로 업로드")
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 _tmp / 출력 사본을 삭제하고 재파생 (원본은 안 건드림)",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="중간 _tmp 사본을 남긴다 / keep the _tmp copy",
    )
    return parser.parse_args()


def lerobot_home():
    """lerobot 데이터셋 캐시 루트 ($HF_LEROBOT_HOME, 기본 ~/.cache/huggingface/lerobot)."""
    return Path(
        os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")
    )


def main():
    arguments = parse_arguments()
    source_repo_id = arguments.repo_id
    feature_name = arguments.feature
    tmp_repo_id = arguments.tmp_repo_id or source_repo_id + "_tmp"
    output_repo_id = arguments.out_repo_id or source_repo_id + "_sliced"
    temporary_key = feature_name + "__sliced_tmp"

    # ── --push-only: 재파생 없이 이미 만든 로컬 산출물만 업로드 ──────────────
    if arguments.push_only:
        output_root = lerobot_home() / output_repo_id
        assert output_root.exists(), f"로컬 산출물 없음 / no local build: {output_root}"
        print(f"[push-only] {output_repo_id}  (private={arguments.private})")
        LeRobotDataset(output_repo_id, root=output_root).push_to_hub(
            private=arguments.private
        )
        print("[push-only] done.")
        return

    # ── 재파생 가드: 기존 파생 사본이 있으면 --force 없이는 중단 ─────────────
    for repo_id in (tmp_repo_id, output_repo_id):
        derived_root = lerobot_home() / repo_id
        if derived_root.exists():
            if arguments.force:
                print(f"[force] 기존 파생 사본 삭제 / rm existing copy {derived_root}")
                shutil.rmtree(derived_root, ignore_errors=True)
            else:
                raise SystemExit(
                    f"이미 존재 / already exists: {derived_root}\n"
                    f"  - 이미 만든 걸 업로드 / push existing build:  --push-only\n"
                    f"  - 지우고 재파생 / delete and rebuild:         --force"
                )

    # ── 원본 로드 + 유지/드롭할 차원 결정 ────────────────────────────────────
    print(f"[load] {source_repo_id}")
    source_dataset = LeRobotDataset(source_repo_id)  # 없으면 다운로드. 읽기 전용.
    feature_info = source_dataset.meta.features[feature_name]
    dimension_count = int(feature_info["shape"][0])
    dtype = feature_info.get("dtype", "float32")
    original_names = feature_info.get("names")
    print(
        f"[check] {feature_name}: dim={dimension_count}, dtype={dtype}, names={original_names}"
    )

    if arguments.keep_first is not None:
        assert (
            0 < arguments.keep_first < dimension_count
        ), f"--keep-first 는 1..{dimension_count - 1} 범위여야 함"
        keep_indices = list(range(arguments.keep_first))
    else:
        drop_indices = sorted(
            {
                int(index)
                for index in arguments.drop_indices.split(",")
                if index.strip() != ""
            }
        )
        assert drop_indices and all(
            0 <= index < dimension_count for index in drop_indices
        ), f"--drop-indices 는 0..{dimension_count - 1} 범위여야 함"
        keep_indices = [
            index for index in range(dimension_count) if index not in set(drop_indices)
        ]

    assert keep_indices, "유지할 차원이 없음 / no dimensions would remain"
    kept_dimension_count = len(keep_indices)
    dropped_indices = [
        index for index in range(dimension_count) if index not in set(keep_indices)
    ]
    keep_index_array = np.array(keep_indices)
    kept_names = (
        [original_names[index] for index in keep_indices]
        if isinstance(original_names, list)
        else None
    )
    dropped_label = (
        f"  ({[original_names[index] for index in dropped_indices]} dropped)"
        if isinstance(original_names, list)
        else ""
    )
    print(
        f"[plan] keep {kept_dimension_count} dims {keep_indices}  |  drop {dropped_indices}{dropped_label}"
    )

    def slice_feature(row_dict, episode_index, frame_index):
        # per-frame 콜백: 유지할 인덱스만 남긴다 / keep only the retained indices
        return np.asarray(row_dict[feature_name], dtype=dtype)[keep_index_array]

    # ── (1) 슬라이스한 feature 를 임시 키로 추가 (새 사본 _tmp) ───────────────
    print(
        f"[step 1] add_features -> {tmp_repo_id}  ({temporary_key} = {kept_dimension_count}-dim)"
    )
    tmp_dataset = add_features(
        source_dataset,
        features={
            temporary_key: (
                slice_feature,
                {"dtype": dtype, "shape": (kept_dimension_count,), "names": kept_names},
            )
        },
        repo_id=tmp_repo_id,
    )

    # ── (2) 원본 feature 제거 (새 사본 -> 출력) ──────────────────────────────
    print(f"[step 2] remove_feature {feature_name} -> {output_repo_id}")
    output_dataset = remove_feature(
        tmp_dataset, feature_names=feature_name, repo_id=output_repo_id
    )
    output_root = Path(output_dataset.root)
    print(f"[step 2] output root = {output_root}")

    # ── (3) 수동 rename: parquet 컬럼 + info.json 키 (사본에만 하므로 원본 무위험) ──
    parquet_files = glob.glob(
        str(output_root / "data" / "**" / "*.parquet"), recursive=True
    )
    assert (
        parquet_files
    ), f"data parquet 을 못 찾음 / no data parquet under {output_root / 'data'}"
    print(
        f"[step 3] rename column in {len(parquet_files)} parquet file(s): {temporary_key} -> {feature_name}"
    )
    for parquet_path in parquet_files:
        table = pd.read_parquet(parquet_path)
        assert (
            temporary_key in table.columns
        ), f"{temporary_key} 컬럼 없음 / missing in {parquet_path}"
        table.rename(columns={temporary_key: feature_name}).to_parquet(
            parquet_path, index=False
        )

    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert temporary_key in info["features"], f"info.json 에 {temporary_key} 없음"
    info["features"][feature_name] = info["features"].pop(temporary_key)
    info_path.write_text(json.dumps(info, indent=4))
    print("[step 3] info.json feature key renamed")

    # ── (4) stats 재계산 (>= 0.5.0) ──────────────────────────────────────────
    print("[step 4] recompute_stats ...")
    recompute_stats(LeRobotDataset(output_repo_id, root=output_root))

    # ── 검증 / verify ────────────────────────────────────────────────────────
    verify_dataset = LeRobotDataset(output_repo_id, root=output_root)
    verify_feature = verify_dataset.meta.features[feature_name]
    verify_stats = verify_dataset.meta.stats.get(feature_name, {})
    print("\n[verify] ---------------------------------------------------------")
    print(
        f"  {feature_name} shape = {verify_feature['shape']}   (expected: ({kept_dimension_count},))"
    )
    print(f"  {feature_name} names = {verify_feature.get('names')}")
    if "mean" in verify_stats:
        mean_values = np.asarray(verify_stats["mean"])
        print(
            f"  stats mean len  = {mean_values.size}, has NaN = {bool(np.isnan(mean_values).any())}"
        )
    print(f"  output root     = {output_root}")
    print("------------------------------------------------------------------\n")
    assert tuple(verify_feature["shape"]) == (
        kept_dimension_count,
    ), "출력 차원 불일치!"

    # ── 중간 _tmp 정리 / cleanup intermediate _tmp ───────────────────────────
    if not arguments.keep_tmp:
        tmp_root = Path(tmp_dataset.root)
        if tmp_root.exists() and tmp_root != output_root:
            print(f"[cleanup] rm {tmp_root}")
            shutil.rmtree(tmp_root, ignore_errors=True)

    # ── (5) 업로드 / push (선택) ─────────────────────────────────────────────
    if arguments.push:
        print(f"[push] {output_repo_id}  (private={arguments.private})")
        verify_dataset.push_to_hub(private=arguments.private)
        print("[push] done.")
    else:
        print(f"[done] local build: {output_root}")
        print("       검증 후 업로드 / verify, then push with:  --push-only")


if __name__ == "__main__":
    main()
