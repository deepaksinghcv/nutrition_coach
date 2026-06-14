"""Single-image inference. Run the base model or a fine-tuned LoRA adapter.

Usage:
    python -m src.infer path/to/food.jpg
    python -m src.infer path/to/food.jpg --adapter adapters/nutrition5k-lora-adapter
"""

import argparse

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from . import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter dir")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen3VLForConditionalGeneration.from_pretrained(config.MODEL_ID, dtype=torch.bfloat16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": config.PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
