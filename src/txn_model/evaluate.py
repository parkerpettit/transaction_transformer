# import torch

# def evaluate(
#     model: torch.nn.Module,
#     loader: torch.utils.data.DataLoader,
#     vocab_sizes: list[int],
#     criterion_cls: torch.nn.Module,
#     criterion_reg: torch.nn.Module,
#     device: torch.device,
# ) -> tuple[float, float, float, list[float]]:
#     """
#     Runs one full pass over `loader` without gradients.
#     Returns (avg_cls_loss, avg_reg_loss, avg_total_loss, accuracies),
#     where accuracies[i] is token-level accuracy for categorical feature i.
#     """
#     model.eval()
#     total_cls, total_reg, total_tokens = 0.0, 0.0, 0
#     correct = [0] * len(vocab_sizes)
#     counts  = [0] * len(vocab_sizes)


#     with torch.no_grad():
#         for batch in loader:
#             # slice inputs / targets
#             inp_cat  = batch["cat"][:, :-1].to(device)
#             inp_cont = batch["cont"][:, :-1].to(device)
#             pad_in   = batch["pad_mask"][:, :-1].to(device).bool()
#             tgt_cont = batch["cont"][:, 1:].to(device)

#             # forward
#             logits_list, preds_cont = model(inp_cat, inp_cont, padding_mask=pad_in)

#             # classification loss + accuracy
#             cls_loss = 0.0
#             mask  = ~batch["pad_mask"][:, 1:].to(device)   # True = real
#             for i, (logits_i, V_i) in enumerate(zip(logits_list, vocab_sizes)):
#                 tgt_i = batch["cat"][:, 1:, i].to(device)     # [B, T]
#                 B, T  = tgt_i.shape
#                 ce    = criterion_cls(logits_i.reshape(B*T, V_i),
#                                       tgt_i.reshape(B*T))
#                 cls_loss += ce

#                 # accumulate accuracy
#                 preds_id = logits_i.argmax(dim=-1)            # [B, T]
#                 real_preds = preds_id[mask]
#                 real_tgts  = tgt_i[mask]
#                 correct[i] += (real_preds == real_tgts).sum().item()
#                 counts [i] += real_tgts.numel()

#             cls_loss /= len(logits_list)

#             # regression loss
#             flat_active = mask.reshape(-1)
#             reg_loss = criterion_reg(
#                 preds_cont.reshape(-1, preds_cont.size(-1))[flat_active],
#                 tgt_cont.reshape(-1, tgt_cont.size(-1))[flat_active]
#             )

#             # accumulate weighted losses
#             n_real = flat_active.sum().item()
#             total_cls    += cls_loss.item() * n_real
#             total_reg    += reg_loss.item() * n_real
#             total_tokens += n_real

#     # compute averages
#     avg_cls   = total_cls / total_tokens
#     avg_reg   = total_reg / total_tokens
#     avg_total = (total_cls * 0.8 + total_reg * 0.2) / total_tokens

#     # compute per-feature accuracies
#     accuracies = [correct[i] / counts[i] for i in range(len(vocab_sizes))]

#     model.train()
#     return avg_cls, avg_reg, avg_total, accuracies


import torch
import torch.nn as nn

def evaluate_binary(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Runs one full pass over `loader` without gradients.
    Returns (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            inp_cat  = batch["cat"][:, :-1].to(device)
            inp_cont = batch["cont"][:, :-1].to(device)
            pad_mask = batch["pad_mask"][:, :-1].to(device).bool()
            labels   = batch["label"].to(device)           # shape: [B] or [B,1]

            # forward
            logits = model(inp_cat, inp_cont, padding_mask=pad_mask)
            
            # handle BCE vs CE
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                logits = logits.view(-1)                   # [B]
                labels_f = labels.float()
                loss = criterion(logits, labels_f)
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                # assume logits shape [B, 2] and labels [B]
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)

            # accumulate
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    model.train()
    return avg_loss, accuracy

