import os
import time
import torch
import torch.nn as nn
from datetime import datetime
from model import TransactionModel
from utils import load_or_initialize_checkpoint, save_checkpoint
from evaluate import evaluate_binary

def train(
    config,
    cat_vocab_mapping,
    cat_features,
    cont_features,
    train_loader,
    val_loader,
):
    """Full training loop for binary fraud classifier, with detailed logging."""
    # 1) Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now()}] Device: {device}")

    # 2) Model, loss, optimizer
    model = TransactionModel(config).to(device)
    print(f"[{datetime.now()}] Initialized model with config: {config}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([  0.5006, 401.1244], device='cuda:0'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 3) Checkpoint loading
    base_path = "/content/drive/MyDrive/summer_urop_25/datasets/txn_checkpoint.pt"
    best_val, start_epoch = load_or_initialize_checkpoint(
        base_path=base_path,
        device=device,
        model=model,
        optimizer=optimizer,
        cat_features=cat_features,
        cont_features=cont_features
    )
    print(f"[{datetime.now()}] Resuming from epoch {start_epoch}, best validation loss = {best_val:.4f}")

    # 4) Training loop
    patience = 5
    wait = 0
    total_epochs = config.total_epochs
    log_interval = 50

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.perf_counter()
        print(f"\n[{datetime.now()}] ===> Starting epoch {epoch+1}/{total_epochs}")

        model.train()
        running_loss = 0.0
        running_samples = 0

        for batch_idx, batch in enumerate(train_loader, 1):
            batch_start = time.perf_counter()

            # --- Move data to device ---
            inp_cat  = batch["cat"][:, :-1].to(device, non_blocking=True)
            inp_cont = batch["cont"][:, :-1].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"][:, :-1].to(device, non_blocking=True).bool()
            labels   = batch["label"].to(device, non_blocking=True)
            # print(
            #     f"[{datetime.now()}] Batch {batch_idx}/{len(train_loader)} shapes: "
            #     f"inp_cat={inp_cat.shape}, inp_cont={inp_cont.shape}, "
            #     f"pad_mask={pad_mask.shape}, labels={labels.shape}"
            # )
            # --- Forward / Backward ---
            logits = model(inp_cat, inp_cont, padding_mask=pad_mask)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Accumulate statistics ---
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

            # --- Per-batch logging ---
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
                batch_time = time.perf_counter() - batch_start
                avg_loss = running_loss / running_samples
                print(
                    f"[{datetime.now()}] Epoch {epoch+1} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Batch Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Batch Time: {batch_time:.2f}s"
                )

        epoch_time = time.perf_counter() - epoch_start
        train_loss = running_loss / running_samples
        print(
            f"[{datetime.now()}] Epoch {epoch+1} finished in {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.4f}"
        )

        # 5) Validation
        val_start = time.perf_counter()
        val_loss, val_acc = evaluate_binary(model, val_loader, criterion, device)
        val_time = time.perf_counter() - val_start
        print(
            f"[{datetime.now()}] Validation | "
            f"Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | "
            f"Time: {val_time:.2f}s"
        )

        # 6) Early-stopping & checkpointing
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            wait = 0
            print(f"[{datetime.now()}] New best model (val_loss: {best_val:.4f}), saving checkpoint.")
            save_checkpoint(
                model, optimizer, epoch, best_val,
                base_path, cat_features, cont_features, config
            )
        else:
            wait += 1
            print(f"[{datetime.now()}] No improvement for {wait}/{patience} epochs.")
            if wait >= patience:
                print(f"[{datetime.now()}] Early stopping triggered. Stopping training.")
                break

    print(f"[{datetime.now()}] Training complete. Best validation loss: {best_val:.4f}")
