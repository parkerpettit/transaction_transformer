import logging
import torch
import torch.nn as nn
from txn_model.config import (
    ModelConfig,
    FieldTransformerConfig,
    SequenceTransformerConfig,
    LSTMConfig,
)
from txn_model.model import TransactionModel

logger = logging.getLogger(__name__)


def evaluate_binary(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[int, float]]:
    """Run a full pass over ``loader`` and compute loss and per-class accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct: dict[int, int] = {}
    class_total: dict[int, int] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            logger.debug("Processing eval batch %d", batch_idx)
            inp_cat = batch["cat"][:, :-1].to(device)
            inp_cont = batch["cont"][:, :-1].to(device)
            pad_mask = batch["pad_mask"][:, :-1].to(device).bool()
            labels = batch["label"].to(device)

            logits = model(inp_cat, inp_cont, padding_mask=pad_mask)

            if isinstance(criterion, nn.BCEWithLogitsLoss):
                logits = logits.view(-1)
                labels_f = labels.float()
                loss = criterion(logits, labels_f)
                preds = (torch.sigmoid(logits) > 0.5).long()
            else:
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            matches = preds == labels
            total_correct += matches.sum().item()
            total_samples += batch_size
            for cls in labels.unique().tolist():
                mask = labels == cls
                if cls not in class_correct:
                    class_correct[cls] = 0
                    class_total[cls] = 0
                class_correct[cls] += (preds[mask] == labels[mask]).sum().item()
                class_total[cls] += mask.sum().item()

            logger.debug(
                "Eval batch %d/%d | loss %.4f", batch_idx, len(loader), loss.item()
            )

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    class_acc = {
        cls: (class_correct[cls] / class_total[cls]) if class_total[cls] > 0 else 0.0
        for cls in class_total
    }



    model.train()
    return avg_loss, accuracy, class_acc
