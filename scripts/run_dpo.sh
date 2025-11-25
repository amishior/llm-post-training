#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

python -m llm_post_training.training.train_dpo \
  --config_path configs/dpo_config.yaml
