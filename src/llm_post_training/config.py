import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Any
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class SFTConfig:
    model_name_or_path: str
    output_dir: str
    train_file: str
    eval_file: str

    max_source_length: int = 512
    max_target_length: int = 512
    max_train_samples: Optional[int] = None

    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class DPOConfig:
    model_name_or_path: str
    ref_model_name_or_path: Optional[str]
    output_dir: str
    train_file: str
    eval_file: str

    max_prompt_length: int = 512
    max_answer_length: int = 512
    max_train_samples: Optional[int] = None

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0

    learning_rate: float = 5e-6
    beta: float = 0.1
    loss_type: str = "sigmoid"

    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 3

    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class EvalConfig:
    model_name_or_path: str
    eval_file: str
    max_input_length: int = 768
    max_new_tokens: int = 512
    batch_size: int = 2
    output_file: str = "outputs/eval/reasoning_eval.jsonl"

    keyword_match: bool = True
    exact_match: bool = False


def load_sft_config(path: str) -> SFTConfig:
    data = load_yaml(path)
    return SFTConfig(**data)


def load_dpo_config(path: str) -> DPOConfig:
    data = load_yaml(path)
    return DPOConfig(**data)


def load_eval_config(path: str) -> EvalConfig:
    data = load_yaml(path)
    return EvalConfig(**data)
