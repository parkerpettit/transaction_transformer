import os, time, torch
import torch.nn as nn

from datetime import datetime
from model import TransactionModel
from utils import load_or_initialize_checkpoint, save_checkpoint
from evaluate import evaluate_binary

# def train(
#   config,
#   cat_vocab_mapping,
#   cat_features,
#   cont_features,
#   train_loader,
#   val_loader,
# ):
#   """Full training loop for ``TransactionModel``."""
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   model  = TransactionModel(config).to(device)

#   PAD_ID = 0
#   criterion_cls = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
#   criterion_reg = nn.MSELoss()
#   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#   vocab_sizes = list(cat_vocab_mapping.values())
#   total_desired_epochs = 300
#   model.train()

#   # ── Hyperparams ──
#   patience = 10          # allowed epochs without improvement
#   best_val = float("inf")
#   wait = 0

#   base_path = "txn_checkpoint.pt"

#   best_val, start_epoch = load_or_initialize_checkpoint(
#       base_path=base_path, device=device, model=model,
#       optimizer=optimizer, cat_features=cat_features, cont_features=cont_features
#   )


#   epochs = total_desired_epochs - start_epoch
#   for epoch in range(start_epoch, start_epoch + epochs + 1):

#       running_loss, n_tokens = 0.0, 0
#       total_tokens = 0

#       for batch_idx, batch in enumerate(train_loader, 1):
#           total_tokens += batch["cat"][:, :-1].numel()
#           # --- slice inputs / targets ---
#           inp_cat  = batch["cat"][:, :-1].to(device)
#           inp_cont = batch["cont"][:, :-1].to(device)
#           pad_in   = batch["pad_mask"][:, :-1].to(device).bool()

#           tgt_cont = batch["cont"][:, 1:].to(device)

#           # --- forward ---
#           logits_list, preds_cont = model(inp_cat, inp_cont, padding_mask=pad_in)

#           # --- classification loss over C categorical heads ---
#           cls_loss = 0.0
#           for i, (logits_i, V_i) in enumerate(zip(logits_list, vocab_sizes)):
#               tgt_i = batch["cat"][:, 1:, i].to(device)     # [B, T]
#               B, T  = tgt_i.shape
#               ce    = criterion_cls(logits_i.reshape(B*T, V_i),
#                                     tgt_i.reshape(B*T))
#               cls_loss +=  ce

#           cls_loss /= len(logits_list)

#           # --- regression loss (mask out pads) ---
#           active = (~batch["pad_mask"][:, 1:]).to(device).reshape(-1)
#           reg_loss = criterion_reg(
#               preds_cont.reshape(-1, preds_cont.size(-1))[active],
#               tgt_cont.reshape(-1, tgt_cont.size(-1))[active]
#           )

#           loss = 0.8 * cls_loss + 0.2* reg_loss

#           # --- backward/optim ---
#           optimizer.zero_grad()
#           loss.backward()
#           optimizer.step()

#           # --- logging ---
#           running_loss += loss.item() * active.sum().item()
#           n_tokens     += active.sum().item()

#       val_cls, val_reg, val_total, accs = evaluate(
#       model, val_loader, vocab_sizes,
#       criterion_cls, criterion_reg, device)

#       print(
#       f"Epoch {epoch} / {start_epoch + epochs} \n"
#       f"  ├─ Validation  | total: {val_total:.4f} | cls: {val_cls:.4f} | reg: {val_reg:.4f}\n"
#       f"  └─ Training    | total: {(running_loss / n_tokens):.4f} | cls: {cls_loss:.4f} | reg: {reg_loss:.4f}\n"
#       )
#       for i, acc in enumerate(accs):
#         print(f"{cat_features[i]} accuracy: {acc:.2%}")
#           # ——— early-stopping logic ———


#       if val_total < best_val - 1e-4:      # a tiny threshold to avoid noise
#           best_val = val_total
#           wait     = 0
#           # save full checkpoint
#           save_checkpoint(model, optimizer, epoch, best_val, base_path, cat_features, cont_features, config)
#       else:
#           wait += 1
#           print(f" No improvement for {wait}/{patience} epochs.")
#           if wait >= patience:
#               print("Early stopping triggered.")
#               break













import torch
import torch.nn as nn

def train(
    config,
    cat_vocab_mapping,
    cat_features,
    cont_features,
    train_loader,
    val_loader,
):
    """Full training loop for binary fraud classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TransactionModel(config).to(device)

    # single binary criterion
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # checkpoint setup (same path, same helper)
    base_path = "txn_checkpoint.pt"
    best_val, start_epoch = load_or_initialize_checkpoint(
        base_path=base_path,
        device=device,
        model=model,
        optimizer=optimizer,
        cat_features=cat_features,
        cont_features=cont_features
    )

    patience = 10
    wait = 0
    total_epochs = config.total_epochs  # e.g. 300

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for batch in train_loader:
            inp_cat  = batch["cat"][:, :-1].to(device)
            inp_cont = batch["cont"][:, :-1].to(device)
            pad_mask = batch["pad_mask"][:, :-1].to(device).bool()
            labels   = batch["label"].to(device).float()   # shape [B]

            logits = model(inp_cat, inp_cont, padding_mask=pad_mask).view(-1)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size    = labels.size(0)
            running_loss += loss.item() * batch_size
            running_samples += batch_size

        train_loss = running_loss / running_samples

        # validation
        val_loss, val_acc = evaluate_binary(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{total_epochs}  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val Acc: {val_acc:.2%}"
        )

        # early-stopping & checkpointing
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            wait = 0
            save_checkpoint(
                model, optimizer, epoch, best_val,
                base_path, cat_features, cont_features, config
            )
        else:
            wait += 1
            print(f" No improvement for {wait}/{patience} epochs.")
            if wait >= patience:
                print("Early stopping triggered.")
                break

