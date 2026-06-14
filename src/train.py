"""QLoRA fine-tuning of Qwen3-VL-4B on Food101 or Nutrition5k.

Designed for a CUDA GPU (e.g. Kaggle T4). 4-bit base + LoRA adapters.

Usage:
    python -m src.train --dataset nutrition5k --out adapters/nutrition5k-lora-adapter
    python -m src.train --dataset food101     --out adapters/food101-lora-adapter

Notes for T4 specifically:
  * compute dtype is fp16 (T4 is Turing — bf16 is emulated/slow)
  * fp16=False AND bf16=False in TrainingArguments: the 4-bit base computes in
    fp16 via bitsandbytes internally, and disabling AMP avoids the GradScaler
    crash on bf16 grads ("_amp_foreach_non_finite_check_and_unscale for BFloat16")
  * we pass peft_config to SFTTrainer (don't pre-wrap with get_peft_model) and
    save trainer.model with save_embedding_layers=False -> ~33MB clean adapter
"""

import argparse

import torch
from transformers import (Qwen3VLForConditionalGeneration, AutoProcessor,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

from . import config
from .data import (build_food101_triplets, build_nutrition5k_triplets,
                   VLMDataset, make_collate_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["food101", "nutrition5k"])
    ap.add_argument("--out", required=True, help="output dir for the LoRA adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-food101", type=int, default=500,
                    help="number of Food101 samples (ignored for nutrition5k)")
    ap.add_argument("--n5k-imagery", default=None,
                    help="override Nutrition5k imagery dir (e.g. Kaggle input path)")
    args = ap.parse_args()

    # --- model in 4-bit ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.MODEL_ID, quantization_config=bnb,
        dtype=torch.float16, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    processor = AutoProcessor.from_pretrained(config.MODEL_ID)

    lora = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
    )

    # --- data ---
    if args.dataset == "food101":
        triplets = build_food101_triplets(n_samples=args.n_food101)
    else:
        triplets = build_nutrition5k_triplets(imagery_dir=args.n5k_imagery)
    print(f"{args.dataset}: {len(triplets)} training triplets")

    train_dataset = VLMDataset(triplets, processor)

    # --- train ---
    targs = TrainingArguments(
        output_dir=args.out + "_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=5, save_strategy="no",
        fp16=False, bf16=False,            # see module docstring
        report_to="none", remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model, args=targs, train_dataset=train_dataset,
        data_collator=make_collate_fn(processor), peft_config=lora,
    )
    trainer.train()

    # Save ONLY the adapter (not the full quantized model)
    trainer.model.save_pretrained(args.out, save_embedding_layers=False)
    processor.save_pretrained(args.out)
    print(f"Saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
