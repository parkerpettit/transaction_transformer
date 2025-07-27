# transaction_transformer

An autoregressive Transformer for credit‑card fraud detection. Pretrain on next‑transaction prediction, then fine‑tune for binary fraud classification.

## Features
- **Self‑supervised pretraining** on 24 M transactions (IBM TabFormer dataset)  
- **Two‑stage pipeline**: predict next transaction -> classify fraud  
- **Flexible heads**: MLP or LSTM classifier on top of frozen or unfrozen Transformer  
- **CLI‑driven experiments** with YAML configs (easy hyperparameter tweaks)  
- **Weights & Biases logging** for metrics, ROC/PR curves, confusion matrices, and artifacts  

