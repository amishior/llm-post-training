import argparse
import json
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from llm_post_training.config import load_eval_config, EvalConfig
from llm_post_training.utils.logging_utils import get_logger
from llm_post_training.utils.seed import set_seed

logger = get_logger(__name__)


def load_eval_data(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data


def keyword_score(pred: str, ref: str) -> float:
    # 非严格，只做个 toy 版本；你后续可以替换为更复杂的 evaluator
    pred_l = pred.lower()
    ref_l = ref.lower()
    if not ref_l.strip():
        return 0.0
    hits = 0
    for token in ref_l.split():
        if token in pred_l:
            hits += 1
    return hits / max(1, len(ref_l.split()))


def exact_score(pred: str, ref: str) -> float:
    return 1.0 if pred.strip() == ref.strip() else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    cfg: EvalConfig = load_eval_config(args.config_path)
    set_seed(42)

    logger.info(f"Loading model from {cfg.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype="auto",
    )
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    eval_data = load_eval_data(cfg.eval_file)
    logger.info(f"Loaded {len(eval_data)} eval samples")

    gen_config = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )

    outputs = []
    scores_keyword = []
    scores_exact = []

    for i, sample in enumerate(eval_data):
        inp = sample["input"]
        ref = sample.get("reference", "")

        inputs = tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_input_length,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                generation_config=gen_config,
            )
        gen_text = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        item = {
            "id": sample.get("id", str(i)),
            "input": inp,
            "reference": ref,
            "prediction": gen_text,
        }

        if cfg.keyword_match and ref:
            k_score = keyword_score(gen_text, ref)
            item["keyword_score"] = k_score
            scores_keyword.append(k_score)
        if cfg.exact_match and ref:
            e_score = exact_score(gen_text, ref)
            item["exact_score"] = e_score
            scores_exact.append(e_score)

        outputs.append(item)

        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i+1}/{len(eval_data)} samples")

    logger.info(f"Saving eval results to {cfg.output_file}")
    import os
    os.makedirs(os.path.dirname(cfg.output_file), exist_ok=True)
    with open(cfg.output_file, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    if scores_keyword:
        logger.info(f"Avg keyword score: {sum(scores_keyword)/len(scores_keyword):.4f}")
    if scores_exact:
        logger.info(f"Avg exact match: {sum(scores_exact)/len(scores_exact):.4f}")

    logger.info("Eval done.")


if __name__ == "__main__":
    main()
