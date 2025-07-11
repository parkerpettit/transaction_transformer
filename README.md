# transaction_transformer
A transformer that predicts the next transaction for a user

## Self-supervised Pre-training

The project supports an auto-regressive pre-training stage. Run

```bash
python scripts/ar_pretrain.py --epochs 20 --batch_size 512 --lr 1e-4
```

to train the backbone in a self-supervised manner. Afterwards fine-tune the
fraud classifier using `main.py`, which automatically loads
`pretrained_backbone.pt` when present. Hyperparameters should remain the same
between the two steps.

## Data Path Configuration

The code expects the raw dataset and intermediate files in a local data
directory. The location can be configured in one of three ways:

1. **Environment variable:** set `TXN_DATA_DIR` to the directory containing
   `tabformer.feather`.
2. **Config file:** create `~/.txn_data_config.json` or
   `~/.txn_data_config.yaml` with a key `data_dir` pointing to your data
   directory.
3. **Default:** if neither of the above is present, the relative `data/`
   directory under the project root is used.

Paths such as `processed_data.pt` and model checkpoints are stored in this
directory as well. Keeping the data outside the repository ensures large files
are not accidentally committed.
