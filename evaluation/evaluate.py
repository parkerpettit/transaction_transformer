from typing import List, Dict, Tuple, Optional, Any

from click import progressbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import wandb
from data.dataset import slice_batch, EmbeddingMemmapDataset
from utils.masking import create_field_and_row_mask


def show_sample_predictions(
    model: nn.Module,
    inp_cat: torch.Tensor,
    inp_cont: torch.Tensor,
    tgt_cat: torch.Tensor,
    tgt_cont: torch.Tensor,
    qtarget: Optional[torch.Tensor],
    cat_features: List[str],
    cont_features: List[str],
    vocab_sizes: List[int],
    cont_vocab_sizes: Dict[str, int],
    mode: str,
    device: torch.device,
    encoders: Optional[Dict[str, Dict[str, Any]]] = None,
    qparams: Optional[Dict[str, Any]] = None
):
    """
    Show predictions vs targets for the first sample in a batch.

    Args:
        model: The model to evaluate
        inp_cat: Categorical inputs
        inp_cont: Continuous inputs  
        tgt_cat: Categorical targets
        tgt_cont: Continuous targets
        qtarget: Quantized targets
        cat_features: List of categorical feature names
        cont_features: List of continuous feature names
        vocab_sizes: List of vocabulary sizes for categorical features
        cont_vocab_sizes: Dictionary of vocabulary sizes for continuous features
        mode: Training mode ("ar" or "masked")
        device: Device to run on
        encoders: Optional encoders for decoding categorical values
        qparams: Optional quantization parameters for decoding continuous values
    """
    # Plan for 200 columns of width
    total_width = 140
    feature_col_width = 20
    expected_col_width = 50
    predicted_col_width = 50
    correct_col_width = 20

    print(f"\n{'='*total_width}")
    print(f"SAMPLE PREDICTIONS vs TARGETS (Mode: {mode.upper()})")
    print(f"{'='*total_width}")

    # Get model predictions
    with torch.no_grad():
        if mode == "ar":
            # AR mode: predict next timestep
            logits = model(inp_cat, inp_cont, mode="ar")  # (B, total_vocab_size)
            logits = logits[0:1]  # Take first sample
        else:
            # MLM mode: predict masked tokens
            # Create mask for evaluation
            total_features = len(cat_features) + len(cont_features)
            mask = create_field_and_row_mask(
                batch_size=inp_cat.shape[0], 
                seq_len=inp_cat.shape[1],
                num_features=total_features,
                field_mask_prob=0.15, 
                row_mask_prob=0.10,
                device=device
            )
            model_output = model(inp_cat, inp_cont, mode="masked", mask=mask)  # (B, L, total_vocab_size) or (num_masked, total_vocab_size)
            
            # Handle tuple return from sparse MLM mode
            if isinstance(model_output, tuple):
                logits, masked_positions = model_output
                if len(logits) == 0:
                    print("  Note: No masked positions found in this sample")
                    return
                # For sparse mode, show predictions for masked positions
                print("  Note: MLM mode with sparse computation - showing masked positions only")
                
                # Get all feature vocab sizes
                all_vocab_sizes = list(vocab_sizes)
                if cont_vocab_sizes:
                    all_vocab_sizes.extend(list(cont_vocab_sizes.values()))

                # Get all targets
                if qtarget is not None:
                    all_targets = torch.cat([tgt_cat, qtarget], dim=2)  # (B, L, total_features)
                else:
                    all_targets = tgt_cat
                all_targets = all_targets[0:1]  # Take first sample

                # Process each feature for masked positions
                start_idx = 0
                feature_idx = 0

                header = (
                    f"{'Feature':<{feature_col_width}}"
                    f"{'Expected':<{expected_col_width}}"
                    f"{'Predicted':<{predicted_col_width}}"
                    f"{'Correct':<{correct_col_width}}"
                )
                print(header)
                print("-" * total_width)

                # Since we have logits for masked positions, all features at those positions were masked
                # Show predictions for the first few masked positions
                num_masked = len(masked_positions)
                print(f"  Number of masked positions: {num_masked}")
                
                # Show predictions for first few masked positions
                max_to_show = min(5, num_masked)  # Show at most 5 masked positions
                
                for pos_idx in range(max_to_show):
                    batch_idx = masked_positions[pos_idx, 0].item()
                    seq_idx = masked_positions[pos_idx, 1].item()
                    
                    print(f"  --- Masked Position {pos_idx + 1} (Batch {batch_idx}, Seq {seq_idx}) ---")
                    
                    # Reset start_idx for each position
                    start_idx = 0
                    feature_idx = 0
                    
                    # Categorical features
                    for i, feat_name in enumerate(cat_features):
                        vocab_len = vocab_sizes[i]
                        
                        # Extract logits for this feature at this masked position
                        feature_logits = logits[pos_idx, start_idx:start_idx+vocab_len]  # (vocab_len,)
                        feature_target = all_targets[0, seq_idx, i]  # scalar
                        
                        predicted_idx = feature_logits.argmax().item()
                        target_idx = feature_target.item()
                        
                        # Skip if target_idx is -100 (ignored position in new masking format)
                        if target_idx == -100:
                            continue
                            
                        is_correct = predicted_idx == target_idx

                        # Try to decode values if encoders are available
                        if encoders and feat_name in encoders:
                            inv_map = encoders[feat_name]["inv"]
                            expected_val = inv_map[target_idx] if target_idx < len(inv_map) else f"ID_{target_idx}"
                            predicted_val = inv_map[predicted_idx] if predicted_idx < len(inv_map) else f"ID_{predicted_idx}"
                        else:
                            expected_val = f"ID_{target_idx}"
                            predicted_val = f"ID_{predicted_idx}"

                        status = "Yes" if is_correct else "No"
                        print(
                            f"    {feat_name:<{feature_col_width-4}}"
                            f"{expected_val:<{expected_col_width}}"
                            f"{predicted_val:<{predicted_col_width}}"
                            f"{status:<{correct_col_width}}"
                        )

                        start_idx += vocab_len
                        feature_idx += 1

                    # Continuous features (quantized)
                    for i, feat_name in enumerate(cont_features):
                        if feat_name in cont_vocab_sizes:
                            vocab_len = cont_vocab_sizes[feat_name]
                            
                            # Extract logits for this feature at this masked position
                            feature_logits = logits[pos_idx, start_idx:start_idx+vocab_len]  # (vocab_len,)
                            feature_target = all_targets[0, seq_idx, feature_idx]  # scalar
                            
                            predicted_idx = feature_logits.argmax().item()
                            target_idx = feature_target.item()
                            
                            # Skip if target_idx is -100 (ignored position in new masking format)
                            if target_idx == -100:
                                continue
                                
                            is_correct = predicted_idx == target_idx

                            # Try to decode quantized values if qparams are available
                            if qparams and feat_name in qparams:
                                centers = qparams[feat_name]["centers"]
                                expected_val = f"{centers[target_idx]:.2f}" if target_idx < len(centers) else f"Bin_{target_idx}"
                                predicted_val = f"{centers[predicted_idx]:.2f}" if predicted_idx < len(centers) else f"Bin_{predicted_idx}"
                            else:
                                expected_val = f"Bin_{target_idx}"
                                predicted_val = f"Bin_{predicted_idx}"

                            status = "Yes" if is_correct else "No"
                            print(
                                f"    {feat_name:<{feature_col_width-4}}"
                                f"{expected_val:<{expected_col_width}}"
                                f"{predicted_val:<{predicted_col_width}}"
                                f"{status:<{correct_col_width}}"
                            )

                            start_idx += vocab_len
                            feature_idx += 1
                    
                    print()  # Empty line between positions
                
                if num_masked > max_to_show:
                    print(f"  ... and {num_masked - max_to_show} more masked positions")
                
                return  # Exit after showing sparse predictions
            else:
                # Field-level MLM mode
                logits = model_output
                
                # Show only features that were actually masked
                print("  Note: Field-level MLM mode - showing only masked features")
                
                # Check which features were masked in the first sample
                first_sample_mask = mask[0]  # (L, total_features)
                masked_features = []
                
                header = (
                    f"{'Feature':<{feature_col_width}}"
                    f"{'Expected':<{expected_col_width}}"
                    f"{'Predicted':<{predicted_col_width}}"
                    f"{'Correct':<{correct_col_width}}"
                )
                print(header)
                print("-" * total_width)
                
                start_idx = 0
                feature_idx = 0
                
                # Check categorical features
                for i, feat_name in enumerate(cat_features):
                    vocab_len = vocab_sizes[i]
                    
                    # Check if this feature was masked anywhere in the sequence
                    feature_mask = first_sample_mask[:, i]  # (L,)
                    if feature_mask.sum() > 0:
                        masked_features.append(feat_name)
                        
                        # Find first masked position for this feature
                        masked_positions_feat = feature_mask.nonzero(as_tuple=False)  # (num_masked, 1)
                        first_masked_pos = int(masked_positions_feat[0, 0].item())
                        
                        # Get logits and target for this masked position
                        feature_logits = logits[0, first_masked_pos, start_idx:start_idx+vocab_len]  # (vocab_len,)
                        feature_target = tgt_cat[0, first_masked_pos, i]  # scalar
                        
                        predicted_idx = feature_logits.argmax().item()
                        target_idx = feature_target.item()
                        
                        # Skip if target_idx is -100 (ignored position in new masking format)
                        if target_idx == -100:
                            continue
                            
                        is_correct = predicted_idx == target_idx

                        # Try to decode values if encoders are available
                        if encoders and feat_name in encoders:
                            inv_map = encoders[feat_name]["inv"]
                            expected_val = inv_map[target_idx] if target_idx < len(inv_map) else f"ID_{target_idx}"
                            predicted_val = inv_map[predicted_idx] if predicted_idx < len(inv_map) else f"ID_{predicted_idx}"
                        else:
                            expected_val = f"ID_{target_idx}"
                            predicted_val = f"ID_{predicted_idx}"

                        status = "Yes" if is_correct else "No"
                        print(
                            f"{feat_name:<{feature_col_width}}"
                            f"{expected_val:<{expected_col_width}}"
                            f"{predicted_val:<{predicted_col_width}}"
                            f"{status:<{correct_col_width}}"
                        )

                    start_idx += vocab_len
                    feature_idx += 1

                # Check continuous features (quantized)
                for i, feat_name in enumerate(cont_features):
                    if feat_name in cont_vocab_sizes:
                        vocab_len = cont_vocab_sizes[feat_name]
                        
                        # Check if this feature was masked anywhere in the sequence
                        feature_mask = first_sample_mask[:, feature_idx]  # (L,)
                        if feature_mask.sum() > 0:
                            masked_features.append(feat_name)
                            
                            # Find first masked position for this feature
                            masked_positions_feat = feature_mask.nonzero(as_tuple=False)  # (num_masked, 1)
                            first_masked_pos = int(masked_positions_feat[0, 0].item())
                            
                            # Get logits and target for this masked position
                            feature_logits = logits[0, first_masked_pos, start_idx:start_idx+vocab_len]  # (vocab_len,)
                            if qtarget is not None:
                                feature_target = qtarget[0, first_masked_pos, i]  # scalar
                            else:
                                feature_target = torch.tensor(0)  # fallback
                            
                            predicted_idx = feature_logits.argmax().item()
                            target_idx = feature_target.item()
                            
                            # Skip if target_idx is -100 (ignored position in new masking format)
                            if target_idx == -100:
                                continue
                                
                            is_correct = predicted_idx == target_idx

                            # Try to decode quantized values if qparams are available
                            if qparams and feat_name in qparams:
                                centers = qparams[feat_name]["centers"]
                                expected_val = f"{centers[target_idx]:.2f}" if target_idx < len(centers) else f"Bin_{target_idx}"
                                predicted_val = f"{centers[predicted_idx]:.2f}" if predicted_idx < len(centers) else f"Bin_{predicted_idx}"
                            else:
                                expected_val = f"Bin_{target_idx}"
                                predicted_val = f"Bin_{predicted_idx}"

                            status = "Yes" if is_correct else "No"
                            print(
                                f"{feat_name:<{feature_col_width}}"
                                f"{expected_val:<{expected_col_width}}"
                                f"{predicted_val:<{predicted_col_width}}"
                                f"{status:<{correct_col_width}}"
                            )

                        start_idx += vocab_len
                        feature_idx += 1
                
                if not masked_features:
                    print("  No features were masked in this sample.")
                else:
                    print(f"\n  Total masked features: {len(masked_features)}")
                    print(f"  Masked features: {', '.join(masked_features)}")
                
                return

    # Get all feature vocab sizes
    all_vocab_sizes = list(vocab_sizes)
    if cont_vocab_sizes:
        all_vocab_sizes.extend(list(cont_vocab_sizes.values()))

    # Get all targets
    if mode == "ar":
        if qtarget is not None:
            all_targets = torch.cat([tgt_cat, qtarget], dim=1)  # (B, total_features)
        else:
            all_targets = tgt_cat
        all_targets = all_targets[0:1]  # Take first sample
    else:
        if qtarget is not None:
            all_targets = torch.cat([tgt_cat, qtarget], dim=2)  # (B, L, total_features)
        else:
            all_targets = tgt_cat
        all_targets = all_targets[0:1]  # Take first sample

    # Process each feature
    start_idx = 0
    feature_idx = 0

    header = (
        f"{'Feature':<{feature_col_width}}"
        f"{'Expected':<{expected_col_width}}"
        f"{'Predicted':<{predicted_col_width}}"
        f"{'Correct':<{correct_col_width}}"
    )
    print(header)
    print("-" * total_width)

    # Categorical features
    for i, feat_name in enumerate(cat_features):
        vocab_len = vocab_sizes[i]

        if mode == "ar":
            feature_logits = logits[0, start_idx:start_idx+vocab_len]  # (vocab_len,)
            feature_target = all_targets[0, i]  # scalar
        else:
            # MLM mode: only show predictions for masked positions
            feature_logits = logits[0, :, start_idx:start_idx+vocab_len]  # (L, vocab_len)
            feature_target = all_targets[0, :, i]  # (L,)
            
            # Find masked positions for this feature
            masked_positions = mask[0].nonzero(as_tuple=False)  # (num_masked, 1)
            if len(masked_positions) == 0:
                # No masked positions for this sample - skip
                start_idx += vocab_len
                feature_idx += 1
                continue
                
            # Show prediction for first masked position
            first_masked_idx = int(masked_positions[0, 0].item())
            feature_logits = feature_logits[first_masked_idx]  # (vocab_len,)
            feature_target = feature_target[first_masked_idx]  # scalar

        predicted_idx = feature_logits.argmax().item()
        target_idx = feature_target.item()
        
        # Skip if target_idx is -100 (ignored position in new masking format)
        if target_idx == -100:
            continue
            
        is_correct = predicted_idx == target_idx

        # Try to decode values if encoders are available
        if encoders and feat_name in encoders:
            inv_map = encoders[feat_name]["inv"]
            expected_val = inv_map[target_idx] if target_idx < len(inv_map) else f"ID_{target_idx}"
            predicted_val = inv_map[predicted_idx] if predicted_idx < len(inv_map) else f"ID_{predicted_idx}"
        else:
            expected_val = f"ID_{target_idx}"
            predicted_val = f"ID_{predicted_idx}"

        status = "Yes" if is_correct else "No"
        print(
            f"{feat_name:<{feature_col_width}}"
            f"{expected_val:<{expected_col_width}}"
            f"{predicted_val:<{predicted_col_width}}"
            f"{status:<{correct_col_width}}"
        )

        start_idx += vocab_len
        feature_idx += 1

    # Continuous features (quantized)
    for i, feat_name in enumerate(cont_features):
        if feat_name in cont_vocab_sizes:
            vocab_len = cont_vocab_sizes[feat_name]

            if mode == "ar":
                feature_logits = logits[0, start_idx:start_idx+vocab_len]  # (vocab_len,)
                feature_target = all_targets[0, feature_idx]  # scalar
            else:
                # MLM mode: only show predictions for masked positions
                feature_logits = logits[0, :, start_idx:start_idx+vocab_len]  # (L, vocab_len)
                feature_target = all_targets[0, :, feature_idx]  # (L,)
                
                # Find masked positions for this feature
                masked_positions = mask[0].nonzero(as_tuple=False)  # (num_masked, 1)
                if len(masked_positions) == 0:
                    # No masked positions for this sample - skip
                    start_idx += vocab_len
                    feature_idx += 1
                    continue
                    
                # Show prediction for first masked position
                first_masked_idx = int(masked_positions[0, 0].item())
                feature_logits = feature_logits[first_masked_idx]  # (vocab_len,)
                feature_target = feature_target[first_masked_idx]  # scalar

            predicted_idx = feature_logits.argmax().item()
            target_idx = feature_target.item()
            
            # Skip if target_idx is -100 (ignored position in new masking format)
            if target_idx == -100:
                continue
                
            is_correct = predicted_idx == target_idx

            # Try to decode quantized values if qparams are available
            if qparams and feat_name in qparams:
                centers = qparams[feat_name]["centers"]
                expected_val = f"{centers[target_idx]:.2f}" if target_idx < len(centers) else f"Bin_{target_idx}"
                predicted_val = f"{centers[predicted_idx]:.2f}" if predicted_idx < len(centers) else f"Bin_{predicted_idx}"
            else:
                expected_val = f"Bin_{target_idx}"
                predicted_val = f"Bin_{predicted_idx}"

            status = "Yes" if is_correct else "No"
            print(
                f"{feat_name:<{feature_col_width}}"
                f"{expected_val:<{expected_col_width}}"
                f"{predicted_val:<{predicted_col_width}}"
                f"{status:<{correct_col_width}}"
            )

            start_idx += vocab_len
            feature_idx += 1

    print(f"{'='*total_width}\n")

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cat_features: List[str],          # names in the same order you sliced tgt_cat
    vocab_sizes: List[int],           # List of vocabulary sizes for categorical features
    cont_features: List[str],         # continuous feature names
    cont_vocab_sizes: Dict[str, int], # quantized continuous vocabulary sizes
    crit_cat: nn.Module,             # e.g. nn.CrossEntropyLoss()
    device: torch.device,
    mode: str = "ar",                # "ar" or "mlm"
    show_predictions: bool = False,  # Show predictions for first sample
    encoders: Optional[Dict[str, Dict[str, Any]]] = None,  # For decoding categorical values
    qparams: Optional[Dict[str, Any]] = None,  # For decoding quantized values
) -> Tuple[float, Dict[str, float]]:
    """
    Validate once over `loader`.

    Returns
    -------
    avg_loss : float
        Mean loss (categorical + quantized continuous) over all samples.
    feat_acc : dict[str, float]
        Accuracy per feature (categorical + quantized continuous) (0-1 range).
    """
    model.eval()
    # progress bar format (reuse from pretrain)
    bar_fmt = (
        "{l_bar}{bar:25}| "
        "{n_fmt}/{total_fmt} batches "
        "({percentage:3.0f}%) | "
        "elapsed: {elapsed} | ETA: {remaining} | "
        "{rate_fmt} | "
        "{postfix}"
    )

    total_loss, total_samples = 0.0, 0
    feat_correct = [0] * len(cat_features)
    feat_total   = [0] * len(cat_features)
    
    # Track continuous feature accuracies
    cont_feat_correct = {}
    cont_feat_total = {}
    if cont_vocab_sizes:
        for feat in cont_features:
            if feat in cont_vocab_sizes:
                cont_feat_correct[feat] = 0
                cont_feat_total[feat] = 0
    prog_bar = tqdm(
    loader,
    desc=f"Val-{mode.upper()}",
    unit="batch",
    total=len(loader),
    ncols=200,
    leave=True,
    bar_format=bar_fmt
    )
    
    # Flag to show predictions only once per evaluation
    predictions_shown = False
    
    for batch in prog_bar:
        if mode == "ar":
            # AR mode: use slice_batch to get single timestep targets
            batch_items = slice_batch(batch)
            if len(batch_items) == 6:
                # With quantization targets
                inp_cat, inp_cont, tgt_cat, tgt_cont, qtarget, _ = (t.to(device) for t in batch_items)
            else:
                # Without quantization targets
                inp_cat, inp_cont, tgt_cat, tgt_cont, _ = (t.to(device) for t in batch_items)
                qtarget = None
        else:
            # MLM mode: use full sequences (no slicing)
            inp_cat = batch["cat_input"].to(device)
            inp_cont = batch["cont_input"].to(device)
            tgt_cat = batch["cat_labels"].to(device)  # Full sequence as targets
            tgt_cont = batch["cont_labels"].to(device)
            qtarget = batch.get("qtarget_labels")
            if qtarget is not None:
                qtarget = qtarget.to(device)
            mask = batch.get("mask")  # Get mask from data collator
            if mask is not None:
                mask = mask.to(device)

        # Forward pass with unified head
        if mode in ["masked", "mlm"]:
            # For MLM with new data collator, inputs are already masked
            if mask is not None:
                # New efficient approach: inputs are already masked from data collator
                model_output = model(inp_cat, inp_cont, mode=mode)  # (B, L, total_vocab_size)
                logits = model_output
                masked_positions = None  # Not needed with new approach
            else:
                # Fallback to old approach for compatibility
                total_features = len(cat_features) + len(cont_features)
                mask = create_field_and_row_mask(
                    batch_size=inp_cat.shape[0], 
                    seq_len=inp_cat.shape[1],
                    num_features=total_features,
                    field_mask_prob=0.15, 
                    row_mask_prob=0.10,
                    device=device
                )
                model_output = model(inp_cat, inp_cont, mode=mode, mask=mask)  # (B, L, total_vocab_size) or (num_masked, total_vocab_size)
                
                # Handle tuple return from sparse MLM mode
                if isinstance(model_output, tuple):
                    logits, masked_positions = model_output
                    # For sparse mode, we need to handle the evaluation differently
                    if len(logits) == 0:
                        # No masked positions - skip this batch
                        batch_size = inp_cat.size(0)
                        total_loss += 0.0 * batch_size
                        total_samples += batch_size
                        prog_bar.set_postfix({"loss": "0.0000"})
                        continue
                else:
                    logits = model_output
                    masked_positions = None
        else:
            logits = model(inp_cat, inp_cont, mode=mode)  # (B, total_vocab_size) or (B, L, total_vocab_size)
            masked_positions = None
        
        # Show sample predictions for first batch if requested
        if show_predictions and not predictions_shown:
            show_sample_predictions(
                model=model,
                inp_cat=inp_cat,
                inp_cont=inp_cont,
                tgt_cat=tgt_cat,
                tgt_cont=tgt_cont,
                qtarget=qtarget,
                cat_features=cat_features,
                cont_features=cont_features,
                vocab_sizes=vocab_sizes,
                cont_vocab_sizes=cont_vocab_sizes,
                mode=mode,
                device=device,
                encoders=encoders,
                qparams=qparams
            )
            predictions_shown = True

        # ----- compute loss & per-feature accuracy -----
        # Get all feature vocab sizes (categorical + quantized continuous) - mirror trainer approach
        all_vocab_sizes = list(vocab_sizes)  # categorical vocab sizes
        if cont_vocab_sizes:
            all_vocab_sizes.extend(list(cont_vocab_sizes.values()))
        
        # Get all targets (categorical + quantized continuous) - mirror trainer approach
        if mode == "ar":
            # AR mode: targets are single timestep
            if qtarget is not None:
                all_targets = torch.cat([tgt_cat, qtarget], dim=1)  # (B, total_features)
            else:
                all_targets = tgt_cat  # fallback if no quantized targets
        else:
            # MLM mode: targets are full sequences
            if qtarget is not None:
                all_targets = torch.cat([tgt_cat, qtarget], dim=2)  # (B, L, total_features)
            else:
                all_targets = tgt_cat  # fallback if no quantized targets
        
        # Compute loss for all features uniformly - mirror trainer approach
        if mode == "ar":
            # AR mode: standard loss computation
            total_loss_val = torch.tensor(0.0, device=device, requires_grad=False)
            start_idx = 0
            num_features = 0
            
            # Process all features uniformly (categorical + continuous)
            for i, vocab_len in enumerate(all_vocab_sizes):
                # AR mode: (B, total_vocab_size)
                feature_logits = logits[:, start_idx:start_idx+vocab_len]  # [B, vocab_len]
                feature_targets = all_targets[:, i]                        # [B]
                
                total_loss_val = total_loss_val + crit_cat(feature_logits, feature_targets)
                num_features += 1
                
                # Accuracy computation
                preds_f = feature_logits.argmax(dim=-1)
                
                # Track accuracy for categorical features
                if i < len(cat_features):
                    feat_correct[i] += (preds_f == feature_targets).sum().item()
                    feat_total[i] += feature_targets.numel()
                # Track accuracy for continuous features
                elif qtarget is not None and cont_vocab_sizes:
                    cont_feat_idx = i - len(cat_features)
                    if cont_feat_idx < len(cont_features):
                        feat_name = cont_features[cont_feat_idx]
                        if feat_name in cont_vocab_sizes:
                            cont_feat_correct[feat_name] += (preds_f == feature_targets).sum().item()
                            cont_feat_total[feat_name] += feature_targets.numel()
                
                start_idx += vocab_len
            
            # Average loss across all features - mirror trainer approach
            loss = total_loss_val / len(all_vocab_sizes) if all_vocab_sizes else total_loss_val
        
        elif masked_positions is not None:
            # Old sparse MLM mode - use the existing logic
            total_loss_val = torch.tensor(0.0, device=device, requires_grad=False)
            start_idx = 0
            num_features = 0
            
            # Process all features uniformly (categorical + continuous)
            for i, vocab_len in enumerate(all_vocab_sizes):
                # Sparse MLM - logits are already for masked positions only
                feature_logits = logits[:, start_idx:start_idx+vocab_len]  # (num_masked, vocab_len)
                # Extract targets for masked positions
                batch_indices = masked_positions[:, 0]  # (num_masked,)
                seq_indices = masked_positions[:, 1]    # (num_masked,)
                feature_targets = all_targets[batch_indices, seq_indices, i]  # (num_masked,)
                
                total_loss_val = total_loss_val + crit_cat(feature_logits, feature_targets)
                num_features += 1
                
                # Accuracy computation
                preds_f = feature_logits.argmax(dim=-1)
                
                # Track accuracy for categorical features
                if i < len(cat_features):
                    feat_correct[i] += (preds_f == feature_targets).sum().item()
                    feat_total[i] += feature_targets.numel()
                # Track accuracy for continuous features
                elif qtarget is not None and cont_vocab_sizes:
                    cont_feat_idx = i - len(cat_features)
                    if cont_feat_idx < len(cont_features):
                        feat_name = cont_features[cont_feat_idx]
                        if feat_name in cont_vocab_sizes:
                            cont_feat_correct[feat_name] += (preds_f == feature_targets).sum().item()
                            cont_feat_total[feat_name] += feature_targets.numel()
                
                start_idx += vocab_len
            
            # Average loss across all features
            loss = total_loss_val / len(all_vocab_sizes) if all_vocab_sizes else total_loss_val
        
        else:
            # Field-level MLM mode - use field-level loss function
            from utils.masking import compute_field_level_mlm_loss
            
            loss = compute_field_level_mlm_loss(
                logits=logits,
                targets_cat=tgt_cat,
                targets_cont=qtarget,
                mask=mask,
                vocab_sizes=vocab_sizes,
                cont_vocab_sizes=cont_vocab_sizes,
                cont_features=cont_features,
                criterion=crit_cat
            )
            
            # Compute accuracy for masked fields only
            start_idx = 0
            for i, vocab_len in enumerate(vocab_sizes):
                # Extract mask for this categorical feature
                feature_mask = mask[:, :, i]  # (B, L)
                
                if feature_mask.sum() > 0:
                    # Extract logits and targets for this feature
                    feature_logits = logits[:, :, start_idx:start_idx+vocab_len]  # (B, L, vocab_size)
                    feature_targets = tgt_cat[:, :, i]  # (B, L)
                    
                    # Flatten and extract only masked positions
                    flat_logits = feature_logits.view(-1, vocab_len)  # (B*L, vocab_size)
                    flat_targets = feature_targets.view(-1)  # (B*L,)
                    flat_mask = feature_mask.view(-1)  # (B*L,)
                    
                    # Extract only masked positions
                    masked_logits = flat_logits[flat_mask]  # (num_masked_for_this_feature, vocab_size)
                    masked_targets = flat_targets[flat_mask]  # (num_masked_for_this_feature,)
                    
                    if len(masked_logits) > 0:
                        preds_f = masked_logits.argmax(dim=-1)
                        feat_correct[i] += (preds_f == masked_targets).sum().item()
                        feat_total[i] += masked_targets.numel()
                
                start_idx += vocab_len
            
            # Handle continuous features
            if qtarget is not None and cont_vocab_sizes:
                for i, feat in enumerate(cont_features):
                    if feat in cont_vocab_sizes:
                        vocab_len = cont_vocab_sizes[feat]
                        feature_idx = len(vocab_sizes) + i  # Index in the mask tensor
                        
                        # Extract mask for this continuous feature
                        feature_mask = mask[:, :, feature_idx]  # (B, L)
                        
                        if feature_mask.sum() > 0:
                            # Extract logits and targets for this feature
                            feature_logits = logits[:, :, start_idx:start_idx+vocab_len]  # (B, L, vocab_size)
                            feature_targets = qtarget[:, :, i]  # (B, L)
                            
                            # Flatten and extract only masked positions
                            flat_logits = feature_logits.view(-1, vocab_len)  # (B*L, vocab_size)
                            flat_targets = feature_targets.view(-1)  # (B*L,)
                            flat_mask = feature_mask.view(-1)  # (B*L,)
                            
                            # Extract only masked positions
                            masked_logits = flat_logits[flat_mask]  # (num_masked_for_this_feature, vocab_size)
                            masked_targets = flat_targets[flat_mask]  # (num_masked_for_this_feature,)
                            
                            if len(masked_logits) > 0:
                                preds_f = masked_logits.argmax(dim=-1)
                                cont_feat_correct[feat] += (preds_f == masked_targets).sum().item()
                                cont_feat_total[feat] += masked_targets.numel()
                        
                        start_idx += vocab_len
        
        batch_size   = inp_cat.size(0)
        total_loss    += loss.item() * batch_size
        total_samples += batch_size
        prog_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total_samples
    feat_acc = {
        name: (feat_correct[i] / feat_total[i]) if feat_total[i] else 0.0
        for i, name in enumerate(cat_features)
    }
    
    # Add continuous feature accuracies
    for feat in cont_features:
        if feat in cont_vocab_sizes and feat in cont_feat_total:
            feat_acc[feat] = (cont_feat_correct[feat] / cont_feat_total[feat]) if cont_feat_total[feat] else 0.0
    

    # Compute total accuracy, handling dicts for continuous features
    total_correct = sum(feat_correct) + sum(cont_feat_correct.values())
    total_count = sum(feat_total) + sum(cont_feat_total.values())
    feat_acc["overall_accuracy"] = (total_correct / total_count) if total_count else 0.0


    
    print("\nFeature-wise Accuracy:")
    for name, acc in feat_acc.items():
        print(f"  - {name:<20}: {acc*100:.2f}%")
    print(f"  └- Avg Val Loss: {avg_loss:.4f}\n")

    model.train()           # restore training mode
    return avg_loss, feat_acc








"""
evaluate_binary.py
------------------
Validation helper for the fraud-classification fine-tuning phase.
Returns loss plus a rich metrics dict and takes care of W&B logging:
  * overall accuracy, precision, recall, F1
  * ROC-AUC, PR-AUC
  * class-wise accuracy
  * confusion matrix, ROC curve, PR curve as interactive W&B plots
"""

from typing import Dict, Tuple, Any, List
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import wandb

from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import wandb
import torch

# ---------- Helper: exact thresholds for FPR limits ----------
def thresholds_for_fpr_limits(y_true: np.ndarray,
                              y_score: np.ndarray,
                              fpr_limits: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_limits: one row per requested fpr_limit with chosen threshold and metrics.
      curve: full monotone curve at every unique probability (optional for analysis).
    """
    y_true  = np.asarray(y_true,  dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float32)

    order = np.argsort(-y_score)               # descending scores
    y_sorted = y_true[order]
    scores_sorted = y_score[order]

    # block ends (unique thresholds)
    change = np.empty_like(scores_sorted, dtype=bool)
    change[:-1] = scores_sorted[:-1] != scores_sorted[1:]
    change[-1] = True
    block_idx = np.nonzero(change)[0]

    pos_mask = (y_sorted == 1)
    cum_pos = np.cumsum(pos_mask)
    cum_neg = np.cumsum(~pos_mask)

    total_pos = int(cum_pos[-1])
    total_neg = int(cum_neg[-1])

    tp = cum_pos[block_idx]
    fp = cum_neg[block_idx]
    fn = total_pos - tp
    tn = total_neg - fp

    recall = tp / total_pos if total_pos else np.zeros_like(tp, dtype=float)
    fpr = fp / total_neg if total_neg else np.zeros_like(fp, dtype=float)
    thresholds = scores_sorted[block_idx]

    # fpr is non-decreasing w.r.t. lowering threshold
    df_curve = pd.DataFrame({
        "threshold": thresholds,
        "fpr": fpr,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    })

    results = []
    for limit in fpr_limits:
        idx = np.searchsorted(fpr, limit, side="right") - 1
        if idx >= 0 and fpr[idx] <= limit:
            chosen = idx
        else:
            # no feasible point (other than predicting none) under limit:
            # choose closest; prefer undershoot
            diffs = np.abs(fpr - limit)
            penalty = (fpr > limit).astype(int)
            # lexicographic: (penalty, diff, threshold) -> pick first
            chosen = np.lexsort((thresholds, diffs, penalty))[0]

        results.append({
            "fpr_limit": float(limit),
            "threshold": float(thresholds[chosen]),
            "fpr": float(fpr[chosen]),
            "recall": float(recall[chosen]),
            "tp": int(tp[chosen]),
            "fp": int(fp[chosen]),
            "fn": int(fn[chosen]),
            "tn": int(tn[chosen]),
        })

    return pd.DataFrame(results), df_curve


@torch.no_grad()
def evaluate_binary(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.BCEWithLogitsLoss,
    device: torch.device,
    mode: str,
    class_names: list[str] | None = None,
):
    model.to(device)
    model.eval()
    # progress bar format (reuse from pretrain)
    bar_fmt = (
        "{l_bar}{bar:25}| "
        "{n_fmt}/{total_fmt} batches "
        "({percentage:3.0f}%) | "
        "elapsed: {elapsed} | ETA: {remaining} | "
        "{rate_fmt} | "
        "{postfix}"
    )

    tot_loss = 0.0
    sample_count = 0
    all_probs: list[float] = []
    all_labels: list[int] = []
    prog_bar = tqdm(
    loader,
    desc="Val-CLS",
    unit="batch",
    total=len(loader),
    ncols=200,
    leave=True,
    bar_format=bar_fmt
    )  

    for batch in prog_bar:
        cat_inp, cont_inp, labels = batch["cat"].to(device), batch["cont"].to(device), batch['label'].to(device)
        labels = labels.float()
        logits = model(cat_inp, cont_inp, mode=mode)  # (B)
        
        loss = criterion(logits, labels.float())
        probs = torch.sigmoid(logits)

        batch_size = labels.size(0)
        tot_loss += loss.item() * batch_size
        sample_count += batch_size

        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        prog_bar.set_postfix({"loss": f"{loss.item():.4f}",
                        "pos":  f"{labels.sum().item():.0f}"})
        
    val_loss = tot_loss / sample_count
    labels_np = np.asarray(all_labels, dtype=np.int8)
    probs_np  = np.asarray(all_probs,  dtype=np.float32)
    probas_2d = np.vstack([1 - probs_np, probs_np]).T

    # ---- FPR-constrained thresholds ----
    fpr_limits = [0.01, 0.001, 0.0005, 0.0001]
    df_limits, _ = thresholds_for_fpr_limits(labels_np, probs_np, fpr_limits)

    # Choose one threshold to compute "main" metrics (example: use .1% FPR row)
    main_thr = df_limits.loc[df_limits.fpr_limit == 0.001, "threshold"].item()
    preds_np = (probs_np >= main_thr).astype(int)

    class_labels = class_names or ["non-fraud", "fraud"]

    # Log confusion matrices at each FPR limit with actual achieved FPR
    for _, row in df_limits.iterrows():
        thr = row.threshold
        fpr_lim = row.fpr_limit
        got_fpr = row.fpr
        y_pred_thr = (probs_np >= thr).astype(int)
        cm = confusion_matrix(labels_np, y_pred_thr, labels=[0,1])

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set(title=f"CM @ target FPR≤{fpr_lim*100:.2f}% (thr={thr:.6f})\nAchieved FPR={got_fpr*100:.4f}%, Recall={row.recall*100:.4f}%")
        key = f"confusion_matrix_fpr_{fpr_lim*100:.2f}pct"
        wandb.log({key: wandb.Image(fig)}, commit=True)
        plt.close(fig)
    
    y_pred_thr = (probs_np >= 0.5).astype(int)
    cm = confusion_matrix(labels_np, y_pred_thr, labels=[0,1])

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set(title=f"CM @ threshold=0.5)")
    key = f"confusion_matrix_0.5_threshold"
    wandb.log({key: wandb.Image(fig)}, commit=True)
    plt.close(fig)

    # Scalar metrics at main_thr
    accuracy = (preds_np == labels_np).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average="binary", zero_division="warn"
    )
    class_acc = {
        0: (preds_np[labels_np == 0] == 0).mean(),
        1: (preds_np[labels_np == 1] == 1).mean(),
    }
    try:
        roc_auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels_np, probs_np)
    except ValueError:
        pr_auc = float("nan")

    # ROC / PR curves
    roc_plot = wandb.plot.roc_curve(
        y_true=labels_np.tolist(),
        y_probas=probas_2d.tolist(),
        labels=class_labels,
    )
    pr_plot = wandb.plot.pr_curve(
        y_true=labels_np.tolist(),
        y_probas=probas_2d.tolist(),
        labels=class_labels,
    )
    wandb.log({"roc_curve": roc_plot, "pr_curve": pr_plot}, commit=True)

    # Log scalar metrics + chosen threshold metrics + per-limit table
    # Flatten df_limits rows into a dict (optional)
    limit_logs = {}
    for _, r in df_limits.iterrows():
        tag = f"fpr_{r.fpr_limit:.6f}"
        limit_logs[f"{tag}_thr"] = r.threshold
        limit_logs[f"{tag}_achieved_fpr"] = r.fpr
        limit_logs[f"{tag}_recall"] = r.recall
        limit_logs[f"{tag}_tp"] = r.tp
        limit_logs[f"{tag}_fp"] = r.fp
        limit_logs[f"{tag}_fn"] = r.fn
        limit_logs[f"{tag}_tn"] = r.tn

    wandb.log({
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_roc_auc": roc_auc,
        "val_pr_auc": pr_auc,
        "val_non_fraud_acc": class_acc[0],
        "val_fraud_acc": class_acc[1],
        "main_threshold_fpr01": main_thr,
        **limit_logs
    }, commit=True)

    model.train()
    return val_loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "class_acc_0": class_acc[0],
        "class_acc_1": class_acc[1],
        "thresholds_table": df_limits 
    }
