"""Compare base Qwen3-VL-4B vs Food101 LoRA vs Nutrition5k LoRA on the same
Nutrition5k holdout images (the only data with real measured ground truth).

Loads the base model once and hot-swaps both adapters onto it, so only one 4B
model sits in memory. Prints per-dish calories and per-nutrient MAE.

Usage:
    python -m src.evaluate                 # all holdout dishes
    python -m src.evaluate --n 8           # quick subset
"""

import os
import re
import json
import argparse
import contextlib

import torch
import pandas as pd
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from . import config

MODES = ("base", "food101", "nutrition5k")
NUTRIENTS = [  # (json field, ground-truth csv column, unit)
    ("calories", "total_calories", "kcal"),
    ("protein_g", "total_protein_g", "g"),
    ("carbs_g", "total_carb_g", "g"),
    ("fat_g", "total_fat_g", "g"),
]


def load_models(device):
    base = Qwen3VLForConditionalGeneration.from_pretrained(config.MODEL_ID, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)
    model = PeftModel.from_pretrained(base, str(config.ADAPTERS / "food101-lora-adapter"),
                                      adapter_name="food101")
    model.load_adapter(str(config.ADAPTERS / "nutrition5k-lora-adapter"),
                       adapter_name="nutrition5k")
    return model.to(device).eval(), processor


def predict(model, processor, image, mode, device):
    if mode == "base":
        ctx = model.disable_adapter()
    else:
        model.set_adapter(mode)
        ctx = contextlib.nullcontext()

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": config.PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with ctx, torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def num(pred, field):
    v = pred.get(field) if isinstance(pred, dict) else None
    return v if isinstance(v, (int, float)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit number of dishes")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device} ...")
    model, processor = load_models(device)

    imagery = config.N5K_HOLDOUT_IMAGERY
    if not (imagery.is_dir() and any(imagery.glob("*.png"))):
        raise SystemExit(f"No holdout images in {imagery}. Run: python -m src.download_holdout")

    cols = ["dish_id", "total_calories", "total_mass_g",
            "total_fat_g", "total_carb_g", "total_protein_g"]
    truth = pd.read_csv(config.N5K_METADATA, header=None,
                        usecols=range(6), names=cols).set_index("dish_id")

    test_ids = sorted(f.stem for f in imagery.glob("*.png"))
    if args.n:
        test_ids = test_ids[:args.n]
    print(f"Testing on {len(test_ids)} holdout dishes\n")

    rows = []
    for i, dish_id in enumerate(test_ids, 1):
        img = Image.open(imagery / f"{dish_id}.png").convert("RGB")
        preds = {m: parse_json(predict(model, processor, img, m, device)) for m in MODES}
        row = {"dish": dish_id[-6:], "food101_name": (preds["food101"] or {}).get("food_name")}
        for field, gt_col, _ in NUTRIENTS:
            row[f"truth_{field}"] = round(float(truth.loc[dish_id, gt_col]), 1)
            for m in MODES:
                row[f"{m}_{field}"] = num(preds[m], field)
        rows.append(row)
        print(f"  [{i}/{len(test_ids)}] {dish_id}: truth={row['truth_calories']} "
              f"base={row['base_calories']} food101={row['food101_calories']} "
              f"n5k={row['nutrition5k_calories']}")

    result = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(result[["dish", "truth_calories", "base_calories", "food101_calories",
                  "nutrition5k_calories", "food101_name"]].to_string(index=False))

    print("\n" + "=" * 70)
    print(f"Mean Absolute Error  (n={len(result)} held-out dishes)\n")
    header = f"{'nutrient':10s}" + "".join(f"{m:>14s}" for m in MODES)
    print(header)
    print("-" * len(header))
    for field, _, unit in NUTRIENTS:
        cells = []
        for m in MODES:
            valid = result.dropna(subset=[f"{m}_{field}", f"truth_{field}"])
            mae = (valid[f"{m}_{field}"] - valid[f"truth_{field}"]).abs().mean() \
                if len(valid) else float("nan")
            cells.append(f"{mae:8.1f} {unit}")
        print(f"{field:10s}" + "".join(f"{c:>14s}" for c in cells))


if __name__ == "__main__":
    main()
