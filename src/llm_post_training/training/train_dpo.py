import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig as TRLDPOConfig

from llm_post_training.config import load_dpo_config, DPOConfig
from llm_post_training.data.dpo_dataset import DPODataset
from llm_post_training.utils.logging_utils import get_logger
from llm_post_training.utils.seed import set_seed

logger = get_logger(__name__)


def build_lora_model(model, cfg: DPOConfig):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    cfg = load_dpo_config(args.config_path)
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

    if cfg.ref_model_name_or_path:
        ref_model_name = cfg.ref_model_name_or_path
    else:
        ref_model_name = cfg.model_name_or_path

    logger.info(f"Using ref model: {ref_model_name}")

    # 准备 HF dataset
    from datasets import Dataset as HFDataset

    def to_hf(ds: DPODataset):
        prompts, chosens, rejecteds = [], [], []
        for item in ds:
            prompts.append(item["prompt"])
            chosens.append(item["chosen"])
            rejecteds.append(item["rejected"])
        return HFDataset.from_dict(
            {
                "prompt": prompts,
                "chosen": chosens,
                "rejected": rejecteds,
            }
        )

    train_dataset = DPODataset(cfg.train_file, max_samples=cfg.max_train_samples)
    train_hf = to_hf(train_dataset)

    def formatting_func(batch):
        """
        TRL DPOTrainer 默认期望字段：
        - prompt
        - chosen
        - rejected
        所以这里直接返回即可，也可以在这里加系统指令模板。
        """
        return batch

    training_args = TRLDPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=["tensorboard"],
        beta=cfg.beta,
        loss_type=cfg.loss_type,
    )

    logger.info("Building DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None if cfg.ref_model_name_or_path is None else ref_model_name,
        args=training_args,
        train_dataset=train_hf,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_length=cfg.max_prompt_length + cfg.max_answer_length,
    )

    logger.info("Start DPO training...")
    trainer.train()
    logger.info("Saving DPO model...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
