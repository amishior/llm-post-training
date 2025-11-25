import argparse
from functools import partial

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

from llm_post_training.config import load_sft_config, SFTConfig
from llm_post_training.data.sft_dataset import SFTDataset
from llm_post_training.utils.logging_utils import get_logger
from llm_post_training.utils.seed import set_seed

logger = get_logger(__name__)


def build_lora_model(model, cfg: SFTConfig):
    if not cfg.use_lora:
        return model
    target_modules = cfg.lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    logger.info("Enabled LoRA")
    return model


def tokenize_function(examples, tokenizer, cfg: SFTConfig):
    # 最简单的拼接方式，你后续可以替换成 chat 模板
    texts = []
    for p, r in zip(examples["prompt"], examples["response"]):
        text = p.strip() + "\n" + r.strip()
        texts.append(text)

    tokenized = tokenizer(
        texts,
        max_length=cfg.max_source_length + cfg.max_target_length,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    cfg = load_sft_config(args.config_path)
    set_seed(42)

    logger.info(f"Loading tokenizer & model from {cfg.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype="auto",
    )
    model = build_lora_model(model, cfg)

    logger.info("Loading datasets...")
    train_dataset = SFTDataset(cfg.train_file, max_samples=cfg.max_train_samples)
    eval_dataset = SFTDataset(cfg.eval_file, max_samples=cfg.max_train_samples)

    # HuggingFace Trainer 需要 map-style dataset -> 用 map tokenization
    def collate_fn(features):
        # 使用 tokenizer 的 pad，只是这里我们已经 padding 了
        return {
            k: [f[k] for f in features] for k in features[0].keys()
        }

    # 这里用 dataset.map 之前先把 dataset 转成 dict-of-lists 不太方便，
    # 所以我们走 Trainer 的 `train_dataset` 传入 tokenized 的版本：
    # 更简单一点：手动构建 tokenized dataset（列表），适配 Trainer。
    from datasets import Dataset as HFDataset

    def to_hf(ds: SFTDataset):
        prompts = []
        responses = []
        for item in ds:
            prompts.append(item["prompt"])
            responses.append(item["response"])
        return HFDataset.from_dict({"prompt": prompts, "response": responses})

    train_hf = to_hf(train_dataset)
    eval_hf = to_hf(eval_dataset)

    tokenized_train = train_hf.map(
        lambda ex: tokenize_function(ex, tokenizer, cfg),
        batched=True,
        remove_columns=train_hf.column_names,
    )
    tokenized_eval = eval_hf.map(
        lambda ex: tokenize_function(ex, tokenizer, cfg),
        batched=True,
        remove_columns=eval_hf.column_names,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_dir=f"{cfg.output_dir}/logs",
        report_to=["tensorboard"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Start SFT training...")
    trainer.train()
    logger.info("Saving model...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
