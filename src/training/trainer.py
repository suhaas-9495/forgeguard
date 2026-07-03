"""
ForgeGuard Phase 1 Trainer
"""

from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer

from src.training.model_loader import load_config


def train_phase1(config_path="configs/training_config.yaml"):

    print("=" * 50)
    print("ForgeGuard Phase 1 Training")
    print("=" * 50)

    cfg = load_config(config_path)

    model_name = cfg["model"]["name"]

    print(f"\nLoading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    print("\nModel loaded successfully")

    # --------------------------------------------------
    # LoRA
    # --------------------------------------------------

    lora_cfg = cfg["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )

    print("\nApplying LoRA...")

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print("\nLoading Dataset...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/processed/train.jsonl",
            "eval": "data/processed/eval.jsonl",
        },
    )

    print("\nDataset Loaded Successfully")
    print("Train Samples:", len(dataset["train"]))
    print("Eval Samples:", len(dataset["eval"]))

    print("\nColumns:")
    print(dataset["train"].column_names)

    # --------------------------------------------------
    # Output Directory
    # --------------------------------------------------

    output_dir = cfg["training"]["output_dir"]

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Training Arguments
    # --------------------------------------------------

    print("\nCreating Training Arguments...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,
        report_to="none",
    )

    print("Training Arguments Created")

    # --------------------------------------------------
    # SFT Trainer
    # --------------------------------------------------

    print("\nCreating SFT Trainer...")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
    )

    print("SFT Trainer Created Successfully")

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nStarting Training...")

    trainer.train()

    print("\nTraining Complete!")

    # --------------------------------------------------
    # Save Adapter
    # --------------------------------------------------

    print("\nSaving Adapter...")

    trainer.save_model(output_dir)

    print("\nAdapter Saved Successfully!")
    print(f"\nSaved To: {output_dir}")

    return output_dir


if __name__ == "__main__":

    adapter_path = train_phase1()

    print("\nReturned:")
    print(adapter_path)