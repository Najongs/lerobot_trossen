"""데이터셋 에피소드의 cam_high 프레임을 진행률 0/25/50/75/100% 에서 뽑아 한 줄씩 붙인다 (파일만 읽음).

usage: scene_rows.py out.png root:ep:label [root:ep:label ...]
"""
import glob
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

out = Path(sys.argv[1])
specs = [s.split(":") for s in sys.argv[2:]]
FR = [0.0, 0.25, 0.5, 0.75, 1.0]
W, H = 320, 240
CAM = "cam_high"
cache = {}


def load(root):
    if root not in cache:
        r = Path(root)
        fps = json.load(open(r / "meta/info.json"))["fps"]
        ep = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(r / "meta/episodes/*/*.parquet")))])
        ep = ep.sort_values("episode_index").reset_index(drop=True)
        data = pd.concat(
            [
                pd.read_parquet(p, columns=["episode_index", "frame_index", "action"])
                for p in sorted(glob.glob(str(r / "data/*/*.parquet")))
            ]
        ).sort_values(["episode_index", "frame_index"])
        cache[root] = (r, fps, ep, data)
    return cache[root]


def frames_at(root, e, idxs):
    r, fps, ep, _ = load(root)
    row = ep.iloc[e]
    k = f"videos/observation.images.{CAM}"
    path = r / f"videos/observation.images.{CAM}/chunk-{int(row[k + '/chunk_index']):03d}/file-{int(row[k + '/file_index']):03d}.mp4"
    t0 = float(row[k + "/from_timestamp"])
    want = [t0 + i / fps for i in idxs]
    c = av.open(str(path))
    vs = c.streams.video[0]
    vs.thread_count = 1
    c.seek(int(max(want[0] - 1.0, 0) / vs.time_base), stream=vs)
    got, j, last = [], 0, None
    for f in c.decode(vs):
        last = f
        while j < len(want) and f.time is not None and f.time >= want[j] - 0.5 / fps:
            got.append(f.to_image())
            j += 1
        if j >= len(want):
            break
    while len(got) < len(want):
        got.append(last.to_image())
    c.close()
    return got


sheet = Image.new("RGB", (W * len(FR), H * len(specs)), "black")
d = ImageDraw.Draw(sheet)
for i, (root, e, label) in enumerate(specs):
    e = int(e)
    r, fps, ep, data = load(root)
    a = np.stack(data[data.episode_index == e]["action"].to_numpy())
    n = len(a)
    idxs = [min(int(round(f * (n - 1))), n - 1) for f in FR]
    cum_th = np.degrees(np.cumsum(a[:, 15]) / fps)
    cum_x = np.cumsum(a[:, 14]) / fps
    ims = frames_at(root, e, idxs)
    for j, (im, ix) in enumerate(zip(ims, idxs)):
        sheet.paste(im.resize((W, H)), (j * W, i * H))
        txt = f"{label} ep{e} f{ix}/{n} t={ix / fps:.1f}s th={cum_th[ix]:+.0f} x={cum_x[ix]:+.2f}"
        d.rectangle((j * W, i * H, j * W + W, i * H + 12), fill="black")
        d.text((j * W + 3, i * H + 1), txt, fill="yellow")
    print(f"{label} ep{e}: n={n} fps={fps} th_end={cum_th[-1]:+.1f} x_end={cum_x[-1]:+.2f}")
sheet.save(out)
print("saved", out)
