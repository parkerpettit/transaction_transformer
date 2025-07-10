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
