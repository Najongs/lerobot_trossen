#!/usr/bin/env python
"""Download meta + only the data/video files that hold the chosen episodes (v3.0 layout).

허브에 codebase 버전 태그가 없는 데이터셋은 LeRobotDataset 이 허브에서 못 읽는다. 태그를 다는 건
허브 쓰기라 하지 않고, 필요한 파일만 로컬로 받아 --root 로 읽는다.
usage: fetch_episodes.py <repo_id> <local_root> <n_episodes>   (고르게 뽑는다, predict_chunks 와 같은 규칙)
"""
import glob, sys
import pandas as pd
from huggingface_hub import snapshot_download

rid, root, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
snapshot_download(rid, repo_type="dataset", local_dir=root, allow_patterns=["meta/*"])
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
N = len(eps)
chosen = sorted({int(round(i * (N - 1) / max(1, n - 1))) for i in range(n)})
sel = eps[eps.episode_index.isin(chosen)]
pats = set()
for _, r in sel.iterrows():
    pats.add(f"data/chunk-{int(r['data/chunk_index']):03d}/file-{int(r['data/file_index']):03d}.parquet")
    for c in eps.columns:
        if c.startswith("videos/") and c.endswith("/chunk_index"):
            key = c[len("videos/"):-len("/chunk_index")]
            pats.add(f"videos/{key}/chunk-{int(r[c]):03d}/file-{int(r[f'videos/{key}/file_index']):03d}.mp4")
snapshot_download(rid, repo_type="dataset", local_dir=root, allow_patterns=sorted(pats))
print("episodes", ",".join(map(str, chosen)))
print("files", len(pats))
