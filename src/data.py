"""Dataset construction for both Food101 and Nutrition5k.

Each example becomes an (image, prompt, response_json) triplet, wrapped in a
chat-format torch Dataset. The collator masks the prompt so the model only
learns from the assistant's JSON answer.
"""

import os
import json
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from . import config


def _response_json(food_name, serving, cal, protein, carbs, fat):
    return json.dumps({
        "food_name": food_name,
        "serving_description": serving,
        "calories": cal,
        "protein_g": protein,
        "carbs_g": carbs,
        "fat_g": fat,
    })


def build_food101_triplets(n_samples=500, seed=42):
    """Food101: class name -> synthetic USDA-average nutrition tuple."""
    from datasets import load_from_disk

    ds = load_from_disk(str(config.FOOD101_DIR))["train"]
    label_names = ds.features["label"].names

    random.seed(seed)
    indices = random.sample(range(len(ds)), n_samples)

    triplets = []
    for i in indices:
        s = ds[i]
        name = label_names[s["label"]]
        cal, prot, carbs, fat = config.FOOD101_NUTRITION[name]
        triplets.append({
            "image": s["image"].convert("RGB"),
            "prompt": config.PROMPT,
            "response": _response_json(name.replace("_", " "), "typical serving",
                                       cal, prot, carbs, fat),
        })
    return triplets


def build_nutrition5k_triplets(imagery_dir=None):
    """Nutrition5k: real lab-measured calories/macros per dish."""
    imagery_dir = str(imagery_dir or config.N5K_TRAIN_IMAGERY)
    cols = ["dish_id", "total_calories", "total_mass_g",
            "total_fat_g", "total_carb_g", "total_protein_g"]
    df = pd.read_csv(config.N5K_METADATA, header=None, usecols=range(6), names=cols)

    have = {f[:-4] for f in os.listdir(imagery_dir) if f.endswith(".png")}
    df = df[df["dish_id"].isin(have)].reset_index(drop=True)

    triplets = []
    for _, row in df.iterrows():
        triplets.append({
            "image": Image.open(f"{imagery_dir}/{row['dish_id']}.png").convert("RGB"),
            "prompt": config.PROMPT,
            "response": _response_json(
                "dish", f"{round(row['total_mass_g'])}g serving",
                round(row["total_calories"], 1), round(row["total_protein_g"], 1),
                round(row["total_carb_g"], 1), round(row["total_fat_g"], 1)),
        })
    return triplets


class VLMDataset(Dataset):
    """Wraps triplets as chat messages and precomputes response token lengths
    (used by the collator to mask the prompt cheaply, without a second
    image-processing pass)."""

    def __init__(self, triplets, processor):
        self.triplets = triplets
        self.resp_lengths = [
            processor.tokenizer(t["response"], add_special_tokens=False,
                                return_tensors="pt").input_ids.shape[1]
            for t in triplets
        ]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": t["prompt"]}]},
                {"role": "assistant", "content": t["response"]},
            ],
            "images": [t["image"]],
            "resp_len": self.resp_lengths[idx],
        }


def make_collate_fn(processor, device=None):
    """Build a collator. On CUDA leave device=None (Trainer moves batches);
    on MPS pass device='mps' since batches aren't auto-moved the same way."""

    def collate_fn(examples):
        msgs = [ex["messages"] for ex in examples]
        imgs = [ex["images"][0] for ex in examples]
        resp_lens = [ex["resp_len"] for ex in examples]

        texts = [processor.apply_chat_template(m, tokenize=False,
                                               add_generation_prompt=False) for m in msgs]
        batch = processor(text=texts, images=imgs, padding=True, return_tensors="pt")
        if device:
            batch = batch.to(device)

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100        # mask padding
        for i, rl in enumerate(resp_lens):                 # mask prompt, keep response
            seq_len = batch["attention_mask"][i].sum().item()
            labels[i, :seq_len - rl - 1] = -100
        batch["labels"] = labels
        return batch

    return collate_fn
