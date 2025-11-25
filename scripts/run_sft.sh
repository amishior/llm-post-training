#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

python -m llm_post_training.training.train_sft \
  --config_path configs/sft_config.yaml
