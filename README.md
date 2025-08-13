## Transaction Transformer

Unified tabular transformer (UniTTab-style) for credit-card fraud detection. Pretrain on tabular time series with either MLM (BERT-style) or AR (next-row) objectives, then fine‑tune for binary fraud classification.

### Highlights
- Uniform handling of categorical and continuous features
- Frequency encoding for continuous inputs; discrete bins as targets with neighborhood label smoothing
- Interchangeable pretraining modes: mlm or ar
- Strong W&B artifact-first workflow; local paths supported via config

## Install

```bash
git clone https://github.com/parkerpettit/transaction_transformer.git
cd transaction_transformer
pip install -e .
```

Python >= 3.8. Key deps: torch, pandas, numpy, scikit-learn, wandb, pyarrow/fastparquet.

## Data and Preprocessing

1) Get the TabFormer credit card dataset (`card_transaction.v1.csv`). See the TabFormer repo (`https://github.com/IBM/TabFormer`) and place the file at `data/raw/card_transaction.v1.csv` (configurable).

2) Optionally upload raw CSV to W&B for lineage:
```bash
upload_raw --config pretrain.yaml --artifact-name raw-card-transactions-v1
```

3) Run preprocessing (fits encoders/scaler/binners on LEGIT TRAIN; exports both LEGIT and FULL splits):
```bash
run_preprocessing
# or
python -m transaction_transformer.data.preprocessing.run_preprocessing
```
Outputs are logged as a dataset artifact `preprocessed-card-v1` with:
- train/val/test parquet files and LEGIT-only variants
- `schema.pt`, `encoders.pt`, `binners.pt`, `scaler.pt`
- `preprocess_meta.json`
- Logs: console INFO; DEBUG to `logs/preprocess/*.log` and attached to the artifact

Path conventions (config-driven; do not hardcode):
- Raw: `data/raw/`
- Processed: `data/processed/`
- Pretrained models: `data/models/pretrained/`
- Finetuned models: `data/models/finetuned/`

## Configuration

All defaults live in YAML; CLI is only for quick overrides.
- Pretrain YAML: `src/transaction_transformer/config/pretrain.yaml`
- Finetune YAML: `src/transaction_transformer/config/finetune.yaml`

Important toggles (edit YAML):
- `metrics.wandb`: true/false, `metrics.wandb_project`: "transaction-transformer", `metrics.wandb_entity`: null or your team
- `model.training.model_type`: "mlm" or "ar"
- `model.data.use_local_inputs`: false uses W&B artifacts; true uses local paths
- `model.training.max_batches_per_epoch`: set a small int (e.g., 1) for ~1‑minute sanity runs
- Special IDs: `padding_idx=0`, `mask_idx=1`, `unk_idx=2`, `ignore_idx=-100`

Note on `--config`: when provided to scripts, it can be a filename like `pretrain.yaml` (resolved relative to `src/transaction_transformer/config/`), or an absolute/relative filesystem path.

## Pretraining

Simple: edit YAML then run
```bash
pretrain
```

Quick overrides (examples supported by the CLI):
```bash
# Choose objective and make a short run
pretrain --config pretrain.yaml --model_type ar --batch-size 32 --total-epochs 1

# Equivalent module form
python -m transaction_transformer.modeling.training.pretrain \
  --config pretrain.yaml \
  --model_type mlm \
  --batch-size 32 \
  --total-epochs 1
```

By default, pretraining consumes the LEGIT splits. Artifacts of the form `pretrained-backbone-<mlm|ar>` are versioned; logs are stored under `logs/pretrain/` and attached to the run.

## Fine‑tuning

Simple: edit YAML then run
```bash
finetune
```

Notes
- If `from_scratch: false` (recommended), the script auto‑pulls `pretrained-backbone-<mlm|ar>:best` from W&B (unless `use_local_inputs: true`, in which case it looks for a local export under `model.pretrain_checkpoint_dir`).
- Class imbalance is handled via `positive_weight` (YAML). If set to 1.0, it is computed from labels.

CLI examples supported:
```bash
finetune --config finetune.yaml --batch-size 128 --total-epochs 1
```

Validation metrics (ROC/PR, threshold tables, and confusion matrices) are logged to W&B. Keys that include FPR include a percent sign in the title for clarity.

## Evaluation (saved models)

During training, the best and last checkpoints are exported and logged as artifacts. To validate specific finetuned artifacts locally:
```bash
evaluate_models
```
The default `evaluate_model.py` contains example artifact names; edit that list to compare your versions.

## Model Architecture Summary
- Categorical features: `nn.Embedding(vocab_size, D, padding_idx=0)` with specials 0=PAD, 1=MASK, 2=UNK
- Continuous features: frequency encoding (L=8) -> per‑feature linear to D; NaNs used as mask sentinels and replaced by a learned mask vector
- Field Transformer (intra‑row) -> RowProjector -> Sequence Transformer (inter‑row; causal for AR, bidirectional for MLM) -> RowExpander -> per‑field heads
- Outputs: dict `{field_name: logits (B, L, V_field)}`; losses with `ignore_index=-100` and label smoothing (categorical) or neighborhood smoothing (continuous)

Default dims (see YAML): field D≈72, sequence d_model≈1080, 12 layers, 12 heads; window=10; bins≈100.

## Project Layout
```
transaction_transformer/
  src/transaction_transformer/
    config/                 # YAMLs + ConfigManager
    data/                   # preprocessing, dataset, collators
    modeling/               # backbone, heads, training scripts
    utils/                  # W&B utils, small helpers
  data/{raw,processed}/
  logs/{preprocess,pretrain,finetune}/
  artifacts/               # local cache of W&B downloads/exports
```

## Weights & Biases
- Set `WANDB_API_KEY` in your environment and `wandb login` once
- Disable by setting `metrics.wandb: false` in YAML
- Artifacts used by default when `use_local_inputs: false` (recommended)

