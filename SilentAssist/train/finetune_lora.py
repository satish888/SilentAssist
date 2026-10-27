"""
finetune_lora.py
LoRA fine-tuning of Phi-3 Mini on clinical / AAC vocabulary

Requirements:
    pip install transformers peft bitsandbytes accelerate datasets

Usage:
    python train/finetune_lora.py \
        --base_model microsoft/Phi-3-mini-4k-instruct \
        --output_dir ./lora_clinical \
        --epochs 3

The fine-tuned adapter is saved to --output_dir and loaded by LLMReranker.
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from silentassist.llm_reranker import CLINICAL_VOCAB


def build_training_data(output_path: str):
    """
    Create a JSONL fine-tuning file from the clinical vocabulary.
    Each entry is a completion task: given context, predict the phrase.
    """
    records = []
    for phrase in CLINICAL_VOCAB:
        # Instruction-following format for Phi-3
        records.append({
            "messages": [
                {
                    "role":    "system",
                    "content": (
                        "You are an AAC (Augmentative and Alternative Communication) "
                        "assistant for non-verbal quadriplegic patients. "
                        "Complete the most likely phrase the patient wants to say."
                    ),
                },
                {
                    "role":    "user",
                    "content": f"Complete: {phrase[:max(1, len(phrase)//2)]}",
                },
                {
                    "role":    "assistant",
                    "content": phrase,
                },
            ]
        })

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} training examples to {output_path}")
    return output_path


def finetune(args):
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            TrainingArguments, BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl  import SFTTrainer
        from datasets import load_dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers peft bitsandbytes trl accelerate datasets")
        return

    # Build training file
    data_path = build_training_data("/tmp/silentassist_clinical.jsonl")

    # Quantisation config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config — target Q and V projection matrices
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=data_path, split="train")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tune Phi-3 Mini on clinical vocab")
    parser.add_argument("--base_model",  default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--output_dir",  default="./lora_clinical")
    parser.add_argument("--epochs",      type=int, default=3)
    args = parser.parse_args()
    finetune(args)
