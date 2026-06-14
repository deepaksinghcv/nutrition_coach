"""Salvage a clean LoRA adapter from a full model.safetensors that was saved
with model.save_pretrained() (the wrong save path).

That full save embeds the 4-bit base (needs bitsandbytes/CUDA) plus the trained
LoRA tensors (plain fp16). We extract only the LoRA tensors, remap their keys
to PEFT's format, and write adapter_model.safetensors + adapter_config.json so
the result loads anywhere as base(bf16) + PeftModel.from_pretrained.

Usage:
    python -m src.extract_adapter <src_dir_with_model.safetensors> <out_dir>
"""

import os
import sys
import json

from safetensors import safe_open
from safetensors.torch import save_file

from . import config


def extract(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    src_file = os.path.join(src_dir, "model.safetensors")
    print(f"Reading LoRA tensors from {src_file}")

    adapter_sd = {}
    with safe_open(src_file, framework="pt") as f:
        for k in f.keys():
            if "lora_A" in k or "lora_B" in k or "modules_to_save" in k:
                # src: model.<path>.lora_A.default.weight
                # dst: base_model.model.model.<path>.lora_A.weight
                new_k = "base_model.model." + k.replace(".default", "")
                adapter_sd[new_k] = f.get_tensor(k)
    print(f"Extracted {len(adapter_sd)} tensors")

    save_file(adapter_sd, os.path.join(out_dir, "adapter_model.safetensors"))

    adapter_config = {
        "peft_type": "LORA", "auto_mapping": None,
        "base_model_name_or_path": config.MODEL_ID, "revision": None,
        "task_type": "CAUSAL_LM", "inference_mode": True,
        "r": config.LORA_R, "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT, "fan_in_fan_out": False, "bias": "none",
        "target_modules": config.LORA_TARGET_MODULES, "modules_to_save": None,
        "init_lora_weights": True, "use_rslora": False, "use_dora": False,
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    sz = os.path.getsize(os.path.join(out_dir, "adapter_model.safetensors")) / 1e6
    print(f"Wrote adapter -> {out_dir}/ ({sz:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m src.extract_adapter <src_dir> <out_dir>")
    extract(sys.argv[1], sys.argv[2])
