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

Python >= 3.8. Key deps: torch, pandas, numpy, scikit-learn, wandb, pyarrow, lora_pytorch.

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

Once this is done once, the resulting artifact is uploaded to W&B. In future runs, it will be downloaded from W&B (if the files don't already exist).

## Configuration

All defaults live in YAML; CLI is only for quick overrides.
- Pretrain YAML: `src/transaction_transformer/config/pretrain.yaml`
- Finetune YAML: `src/transaction_transformer/config/finetune.yaml`

Supported config keys (edit YAML):
- `metrics.wandb_project`, `metrics.run_name`, `metrics.run_id`, `metrics.seed`
- `model.mode` (pretrain|finetune), `model.head_type` (mlp|lstm)
- Transformers: `model.field_transformer.*`, `model.sequence_transformer.*`
- Embedding: `model.embedding.*` (emb_dim must match field d_model)
- Classification head: `model.classification.*`
- Training: `model.training.{model_type,batch_size,total_epochs,learning_rate,p_field,p_row,use_amp,num_workers,early_stopping_patience,max_batches_per_epoch,device,positive_weight}`
- Data: `model.data.{use_local_inputs,raw_csv_path,raw_artifact_name,preprocessed_artifact_name,window,stride,num_bins,group_by,include_all_fraud,padding_idx,mask_idx,unk_idx,ignore_idx}`

Note on `--config`: when provided to scripts, it can be a filename like `pretrain.yaml` (resolved relative to `src/transaction_transformer/config/`), or an absolute/relative filesystem path.

Minimal examples:

```yaml
# src/transaction_transformer/config/pretrain.yaml
model:
  mode: "pretrain"
  field_transformer: { d_model: 72, n_heads: 8, depth: 1, ffn_mult: 4, dropout: 0.1 }
  sequence_transformer: { d_model: 1080, n_heads: 12, depth: 12, ffn_mult: 4, dropout: 0.1 }
  embedding: { emb_dim: 72, dropout: 0.1, padding_idx: 0, freq_encoding_L: 8 }
  head: { hidden_dim: 1080, depth: 0, dropout: 0.1 }
  training: { model_type: "mlm", batch_size: 64, total_epochs: 5, learning_rate: 5.0e-5, p_field: 0.15, p_row: 0.10, use_amp: true, num_workers: 4 }
  data: { preprocessed_path: "data/processed/", use_local_inputs: false, raw_csv_path: "data/raw/card_transaction.v1.csv", raw_artifact_name: "raw-card-transactions", preprocessed_artifact_name: "preprocessed-card", window: 10, stride: 5, num_bins: 100, group_by: "User", include_all_fraud: false, padding_idx: 0, mask_idx: 1, unk_idx: 2, ignore_idx: -100 }
metrics: { run_name: pretrain, run_id: null, wandb_project: "transaction-transformer", seed: 42 }
```

```yaml
# src/transaction_transformer/config/finetune.yaml
model:
  mode: "finetune"
  head_type: "lstm"
  field_transformer: { d_model: 72, n_heads: 8, depth: 1, ffn_mult: 4, dropout: 0.1 }
  sequence_transformer: { d_model: 1080, n_heads: 12, depth: 12, ffn_mult: 4, dropout: 0.1 }
  classification: { hidden_dim: 512, depth: 2, dropout: 0.1, output_dim: 1 }
  embedding: { emb_dim: 72, dropout: 0.1, padding_idx: 0, freq_encoding_L: 8 }
  training: { model_type: "mlm", batch_size: 256, total_epochs: 1, learning_rate: 1.0e-4, use_amp: true, num_workers: 4, positive_weight: 9.08, early_stopping_patience: 10, max_batches_per_epoch: 5, resume: false }
  data: { preprocessed_path: "data/processed/", use_local_inputs: false, raw_csv_path: "data/raw/card_transaction.v1.csv", raw_artifact_name: "raw-card-transactions-v1", preprocessed_artifact_name: "preprocessed-card-v1", window: 10, stride: 10, num_bins: 100, group_by: "User", include_all_fraud: true, padding_idx: 0, mask_idx: 1, unk_idx: 2, ignore_idx: -100 }
metrics: { run_name: finetune, run_id: null, wandb_project: "transaction-transformer", seed: 42 }
```

Reproducibility:
- `metrics.seed` is applied to torch, numpy, random, and CUDA seeds in pretrain, finetune, and evaluate.

## Pretraining

Simple: edit YAML then run
```bash
pretrain
```

Quick overrides (examples supported by the CLI):
```bash
# Choose objective and make a short run
pretrain --config pretrain.yaml --model-type ar --batch-size 32 --total-epochs 1

# Equivalent module form
python -m transaction_transformer.modeling.training.pretrain \
  --config pretrain.yaml \
  --model-type mlm \
  --batch-size 32 \
  --total-epochs 1
```

CLI flags supported:
- --config, --model-type (mlm|ar), --head-type (mlp|lstm)
- --batch-size, --learning-rate, --total-epochs, --device
- --field-d-model, --field-n-heads, --field-depth
- --seq-d-model, --seq-n-heads, --seq-depth
- --window, --stride, --num-bins, --p-field, --p-row
- --use-amp/--no-use-amp, --run-name, --seed

By default, pretraining consumes the LEGIT splits. Artifacts are versioned; logs are stored under `logs/pretrain/`.

## Fine‑tuning

Simple: edit YAML then run
```bash
finetune
```

Notes
- The script pulls `pretrain-<mlm|ar>:latest` for finetuning by default.
- Class imbalance is handled via `positive_weight` (YAML). If set to 1.0, it is computed from labels.

CLI examples supported:
```bash
finetune --config finetune.yaml --batch-size 128 --total-epochs 1
```

Validation metrics (ROC/PR, threshold tables, and confusion matrices) are logged to W&B.
<!-- ## Evaluation (saved models)

During training, the best and last checkpoints are exported and logged as artifacts. To validate specific finetuned artifacts locally:
```bash
evaluate_models
```
The default `evaluate_model.py` contains example artifact names; edit that list to compare your versions. -->

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
- Artifacts used by default when `use_local_inputs: false` (recommended)

## LoRA
- LoRA adapters are available via `lora_pytorch`. The finetune script has an example toggle you can adapt.

