# LLM Post-Training Playground

A small but complete project for LLM **post-training**:

- Supervised Fine-tuning (SFT)
- DPO-based preference optimization
- Basic reasoning evaluation

## Structure

- `configs/`: YAML configs for SFT / DPO / eval
- `data/raw/`: Put your JSONL format data here
- `src/llm_post_training/`: Source code

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run SFT

```bash 
scripts/run_sft.sh
```

### 3. Run DPO

```bash 
scripts/run_dpo.sh
```

### 4. Run Eval

```bash 
scripts/run_eval.sh
```

```bash 
llm-post-training/
├── README.md
├── requirements.txt
├── configs
│   ├── sft_config.yaml
│   ├── dpo_config.yaml
│   └── eval_config.yaml
├── data
│   ├── raw
│   │   ├── sft_train.jsonl
│   │   ├── sft_eval.jsonl
│   │   ├── dpo_train.jsonl
│   │   └── eval_reasoning.jsonl
│   └── processed
├── scripts
│   ├── run_sft.sh
│   ├── run_dpo.sh
│   └── run_eval.sh
└── src
    └── llm_post_training
        ├── __init__.py
        ├── config.py
        ├── utils
        │   ├── logging_utils.py
        │   └── seed.py
        ├── data
        │   ├── sft_dataset.py
        │   └── dpo_dataset.py
        ├── training
        │   ├── train_sft.py
        │   └── train_dpo.py
        └── eval
            └── eval_reasoning.py
```