"""Download a clean Nutrition5k holdout set (dishes NOT used for training) so
evaluation measures generalization rather than memorization.

Requires gsutil (Google Cloud SDK) on PATH.

Usage:
    python -m src.download_holdout --n 15
"""

import os
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import config

BUCKET = ("gs://nutrition5k_dataset/nutrition5k_dataset/imagery/"
          "realsense_overhead/{dish_id}/rgb.png")


def download_dish(dish_id, out_dir):
    dst = os.path.join(out_dir, f"{dish_id}.png")
    if os.path.exists(dst):
        return dish_id, 0
    r = subprocess.run(["gsutil", "cp", BUCKET.format(dish_id=dish_id), dst],
                       capture_output=True)
    return dish_id, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="number of holdout dishes")
    args = ap.parse_args()

    out_dir = str(config.N5K_HOLDOUT_IMAGERY)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(config.N5K_METADATA, header=None, usecols=[0], names=["dish_id"])
    trained = {f[:-4] for f in os.listdir(config.N5K_TRAIN_IMAGERY) if f.endswith(".png")}
    candidates = [d for d in df["dish_id"] if d not in trained]
    print(f"{len(candidates)} dishes available outside the training set")

    # over-fetch (some dishes lack an overhead RGB image)
    target = candidates[: args.n * 3]
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(lambda d: download_dish(d, out_dir), target)

    got = [f for f in os.listdir(out_dir) if f.endswith(".png")]
    print(f"Holdout now has {len(got)} images in {out_dir}")


if __name__ == "__main__":
    main()
