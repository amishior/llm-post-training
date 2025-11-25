#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

python -m llm_post_training.eval.eval_reasoning \
  --config_path configs/eval_config.yaml
