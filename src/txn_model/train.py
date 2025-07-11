import os
import time
import logging
from datetime import datetime

from path_utils import get_data_dir

import torch
import torch.nn as nn

from model import TransactionModel
from utils import load_or_initialize_checkpoint, save_checkpoint
from evaluate import evaluate_binary

logger = logging.getLogger(__name__)


def train(
    config,
    cat_vocab_mapping,
    cat_features,
    cont_features,
    train_loader,
    val_loader,
):
    """Training loop for binary fraud classifier with detailed logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    print(f"[Train] Using device {device}")

    model = TransactionModel(config).to(device)
    logger.info("Initialized model with config: %s", config)

    if os.path.exists("pretrained_backbone.pt"):
        logger.info("Loading pretrained backbone")
        state = torch.load("pretrained_backbone.pt", map_location=device)
        model.load_state_dict(state, strict=False)
        for name, param in model.named_parameters():
            if not name.startswith("fraud_head"):
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([0.5006, 401.1244], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    data_dir = get_data_dir()
    base_path = os.path.join(data_dir, "txn_checkpoint.pt")
    best_val, start_epoch = load_or_initialize_checkpoint(
        base_path=base_path,
        device=device,
        model=model,
        optimizer=optimizer,
        cat_features=cat_features,
        cont_features=cont_features,
    )
    logger.info("Resuming from epoch %d with best val %.4f", start_epoch, best_val)
    print(f"[Train] Resuming from epoch {start_epoch} with best val {best_val:.4f}")

    patience = 5
    wait = 0
    total_epochs = config.total_epochs
    log_interval = 50

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.perf_counter()
        logger.info("Starting epoch %d/%d", epoch + 1, total_epochs)
        print(f"[Train] Epoch {epoch + 1}/{total_epochs} start")

        model.train()
        running_loss = 0.0
        running_samples = 0

        for batch_idx, batch in enumerate(train_loader, 1):
            batch_start = time.perf_counter()
            logger.debug("Epoch %d batch %d", epoch + 1, batch_idx)
            inp_cat = batch["cat"][:, :-1].to(device, non_blocking=True)
            inp_cont = batch["cont"][:, :-1].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"][:, :-1].to(device, non_blocking=True).bool()
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(inp_cat, inp_cont, padding_mask=pad_mask, mode="fraud")
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

            if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
                batch_time = time.perf_counter() - batch_start
                avg_loss = running_loss / running_samples
                logger.info(
                    "Epoch %d Batch %d/%d | Batch Loss %.4f | Avg Loss %.4f | Time %.2fs",
                    epoch + 1,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    avg_loss,
                    batch_time,
                )

        epoch_time = time.perf_counter() - epoch_start
        train_loss = running_loss / running_samples
        logger.info(
            "Epoch %d finished in %.2fs | Train Loss %.4f",
            epoch + 1,
            epoch_time,
            train_loss,
        )
        print(f"[Train] Epoch {epoch + 1} done. Train loss {train_loss:.4f}")

        val_start = time.perf_counter()
        val_loss, val_acc, class_acc = evaluate_binary(
            model, val_loader, criterion, device
        )
        val_time = time.perf_counter() - val_start
        logger.info(
            "Validation | Loss %.4f | Acc %.2f%% | Time %.2fs",
            val_loss,
            val_acc * 100,
            val_time,
        )
        for cls, acc in class_acc.items():
            logger.info("Class %s Acc %.2f%%", cls, acc * 100)
        print(f"[Train] Validation loss {val_loss:.4f} acc {val_acc*100:.2f}%")
        for cls, acc in class_acc.items():
            print(f"[Train] Class {cls} acc {acc*100:.2f}%")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            wait = 0
            logger.info("New best model (val_loss %.4f). Saving checkpoint.", best_val)
            print("[Train] New best model, saving checkpoint")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val,
                base_path,
                cat_features,
                cont_features,
                config,
            )
        else:
            wait += 1
            logger.info("No improvement for %d/%d epochs", wait, patience)
            print(f"[Train] No improvement for {wait}/{patience} epochs")
            if wait >= patience:
                logger.info("Early stopping triggered")
                print("[Train] Early stopping triggered")
                break

    logger.info("Training complete. Best validation loss: %.4f", best_val)
    print(f"[Train] Training complete. Best val {best_val:.4f}")
