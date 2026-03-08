# Speaker-Listener RL for Summarization (DSC-Capstone-A10-2)

This repo trains a speaker model (GPT-style LM) using listener-based preferences:
- Speaker: generates candidate summaries.
- Listener: scores candidates against source text with BERTScore.
- Trainer: applies DPO objective to improve speaker outputs.

The current workflow is based on the unified `train_100M` data pipeline. Older CHILDES-only and `K=2` docs are no longer the primary path.

## Project Organization

The codebase is organized as Python modules/scripts (not notebook-driven):
- `data/`: dataset builders and loaders (`*.py` scripts, JSONL assets)
- `training/`: training entrypoints (pretraining/SFT/DPO) and model eval script
- `listener/`: listener/scoring code
- `speaker/`: speaker model components
- `scripts/`: utility runners for generation/pair creation
- `yaml/`: Kubernetes job manifests for cluster runs
- `outputs/`: run artifacts (generated files/checkpoints)

## Environment Recreation

The development environment can be recreated with either Conda or `venv`.

### Option A: Conda (recommended)

```bash
conda create -n dsc-capstone python=3.10 -y
conda activate dsc-capstone
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional (for experiment tracking):

```bash
wandb login
```

Quick install sanity check:

```bash
python -c "import torch, transformers, trl, bert_score; print('env ok')"
```

## Nautilus / Kubernetes Reproduction (Namespace-Based)

This project was run on Nautilus with jobs in a user namespace. Use the manifests in `yaml/`.

### 1) Set kubectl context + namespace

```bash
kubectl config current-context
kubectl config set-context --current --namespace=<your-namespace>
kubectl get ns
```

Note:
- Several manifests hardcode `namespace: dsc-capstone-25-26`; change that to your namespace before applying.
- `yaml/dpo_pvc.yaml` currently uses PVC name `dpo-storage`, while training jobs mount `dpo-full-storage`. Keep these names consistent (edit one side before deploy).

### 2) Create W&B secret in your namespace

```bash
kubectl create secret generic wandb-api-key \
  --from-literal=WANDB_API_KEY=<your_wandb_api_key> \
  -n <your-namespace>
```

### 3) Create persistent volumes

```bash
kubectl apply -f yaml/dpo_pvc.yaml -n <your-namespace>
kubectl apply -f yaml/dpo_sweep_pvc.yaml -n <your-namespace>
```

### 4) Launch jobs

Pretraining/SFT:

```bash
kubectl apply -f yaml/pretraining_job.yaml -n <your-namespace>
```

DPO post-training:

```bash
kubectl apply -f yaml/dpo_posttraining_job.yaml -n <your-namespace>
```

Optional sweep job:

```bash
kubectl apply -f yaml/sweep_job.yaml -n <your-namespace>
```

### 5) Monitor and debug

```bash
kubectl get pods -n <your-namespace>
kubectl get jobs -n <your-namespace>
kubectl logs -f job/dpo-pretraining -n <your-namespace>
kubectl logs -f job/dpo-posttraining -n <your-namespace>
```

Debug pod / checkpoint copy helpers:

```bash
kubectl apply -f yaml/debug_pod.yaml -n <your-namespace>
kubectl apply -f yaml/copy_checkpoint.yaml -n <your-namespace>
```

### 6) Cleanup

```bash
kubectl delete job dpo-pretraining dpo-posttraining -n <your-namespace>
kubectl delete pod pvc-debug pvc-copy -n <your-namespace>
```

## Data Build Instructions

### 1) Build the combined 100M-train passage dataset

```bash
python data/make_all_train_100m.py \
  --train_dir data/train_100M \
  --output_path data/train_100m_passages.jsonl \
  --min_words 20 \
  --target_passage_words 80
```

Expected output:
- `data/train_100m_passages.jsonl`

### 2) (Optional) Build TF-IDF weights for SFT/pretraining pipeline

```bash
python -m data.build_tf_idf \
  --wiki_jsonl data/train_100m_passages.jsonl \
  --out idf_wiki.json
```

Expected output:
- `idf_wiki.json`

## Main Training Path (DPO + Listener)

Run:

```bash
python training/dpo_full_train.py \
  --policy_model gpt2 \
  --input_path data/train_100m_passages.jsonl \
  --output_path outputs/dpo_train100m \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 1e-5 \
  --alpha 0.01 \
  --beta 0.1 \
  --max_length 256 \
  --top_p 0.9 \
  --temperature 0.7 \
  --max_new_tokens 16 \
  --max_pair_similarity 0.85 \
  --max_resample_tries 2 \
  --listener_model_type bert-base-uncased \
  --listener_batch_size 8 \
  --test_size 0.1 \
  --split_seed 42 \
  --run_validation \
  --validation_max_examples 128
```

Output artifacts:
- `outputs/dpo_train100m/checkpoint-*`
- `outputs/dpo_train100m/final_model`

## Test a Trained Model

```bash
python training/test_model.py \
  --model_dir outputs/dpo_train100m/final_model \
  --prompt "Keywords summary. Text: The quick brown fox jumps over the lazy dog. Output:" \
  --max_new_tokens 16
```

## Smoke Test (Small Run)

```bash
python training/dpo_full_train.py \
  --policy_model gpt2 \
  --input_path data/simple_wiki_passages_8k.jsonl \
  --output_path outputs/dpo_smoke \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum 2 \
  --validation_max_examples 16
```

## Reproducibility Notes

- Keep `--split_seed` fixed for deterministic train/validation split.
- Keep model + decode settings fixed (`top_p`, `temperature`, `max_new_tokens`).
- Log the full command line for each run.
- Use the same dependency set from `requirements.txt`.

## Common Issues

- CUDA OOM: lower `--batch_size`, `--listener_batch_size`, or `--max_length`.
- Slow training: GPU is strongly recommended; CPU runs are possible but much slower.
- W&B not logging: run `wandb login` and pass `--wandb_project`/`--wandb_run_name`.
