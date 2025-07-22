#!/usr/bin/env python
"""
Masked‑LM pre‑training for TransactionModel (tabular time‑series).

• Predicts masked categorical IDs + quantised numeric bins.
• One Linear head per field; logits computed **only** on masked rows.
• Saves the encoder backbone to  <data_dir>/bert_backbone.pt
"""
from __future__ import annotations
import argparse, signal, sys, time
from pathlib import Path
from typing  import Dict, List, Any
import torch.nn.functional as F
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto        import tqdm
import wandb

from config       import ModelConfig, TransformerConfig
from data.dataset import TxnDataset, collate_mlm
from model        import TransactionModel
from utils        import load_cfg, merge, load_ckpt, save_ckpt
from data.dataset import get_dataset, collate_mlm

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def graceful_exit(*_):
    print("[Ctrl‑C] Exiting."); sys.exit(0)

@torch.no_grad()
@torch.inference_mode()
def evaluate_mlm(
    model:        nn.Module,
    loader:       DataLoader,
    field_sizes:  List[int],
    field_names:  List[str],
    device:       torch.device,
    *,
    num_numeric:  int = 1,
    eps_num:      float = 0.1,
    win:          int = 5,
) -> Dict[str, Any]:
    """
    Evaluate masked-LM pre-training loss + per-field accuracy
    (masked tokens only).  Mirrors the training logic exactly.
    """
    import torch.nn.functional as F

    model.eval()
    crit_cat  = nn.CrossEntropyLoss(reduction="sum", label_smoothing=0.1)
    num_start = len(field_sizes) - num_numeric        # first numeric field

    IGN = -100
    tot_loss = tot_tok = 0.0
    field_corr = [0] * len(field_sizes)
    field_tok  = [0] * len(field_sizes)

    for batch in loader:
        cat, cont, cbin = [batch[k].to(device) for k in ("cat", "cont", "contbin")]
        B, T, Fc = cat.shape
        Fn       = cbin.size(2)

        # ----- build fresh random masks (same policy as training) ------------
        mcat = (torch.rand_like(cat, dtype=torch.float32) < 0.15) | \
               (torch.rand(B, T, 1, device=device) < 0.10)
        mnum = (torch.rand(B, T, Fn, device=device) < 0.15) | \
               (torch.rand(B, T, 1, device=device) < 0.10)

        cat_labels  = cat .masked_fill(~mcat, IGN)
        cont_labels = cbin.masked_fill(~mnum, IGN)

        # hide inputs at the masked positions
        cat  = cat .masked_fill(mcat, 0)
        cont = cont.masked_fill(mnum, 0.0)

        labels_all = torch.cat((cat_labels, cont_labels), dim=-1)   # (B,T,F)
        mask_all   = labels_all.ne(IGN)

        # -------- forward pass ----------------------------------------------
        seq_flat  = model.encode(cat, cont)                         # (B*T, d)
        F_tot     = labels_all.size(-1)
        flat_lab  = labels_all.view(-1, F_tot)
        flat_mask = mask_all .view(-1, F_tot)

        # -------- per‑field loss / accuracy -------------------------------
        for f, V in enumerate(field_sizes):
            rows_f = flat_mask[:, f]
            if not rows_f.any():
                continue

            logits_f = model.mlm_head_layers[f](seq_flat[rows_f])   # (N,V)
            labels_f = flat_lab[rows_f, f]                         # (N,)

            if f < num_start:                                      # categorical
                loss = crit_cat(logits_f, labels_f)

            else:                                                  # numeric bin
                if eps_num == 0:
                    loss = crit_cat(logits_f, labels_f)
                else:
                    N = labels_f.size(0)
                    soft = torch.zeros((N, V), device=device, dtype=logits_f.dtype)
                    idx  = labels_f.unsqueeze(1)                   # (N,1)

                    p_val = eps_num / (2 * win)
                    offsets = torch.arange(-win, win + 1, device=device)
                    neigh   = torch.clamp(idx + offsets, 0, V - 1)  # (N,2w+1)

                    # centre + neighbours
                    soft.zero_()
                    soft.scatter_add_(1, neigh,
                                       torch.full_like(neigh, p_val, dtype=soft.dtype))
                    soft.scatter_(1, idx, 1.0 - eps_num)

                    loss = -(soft * F.log_softmax(logits_f, 1)).sum()

            # bookkeeping
            tot_loss += loss.item()
            ntok      = labels_f.numel()
            tot_tok  += ntok

            preds = logits_f.argmax(1)
            field_corr[f] += (preds == labels_f).sum().item()
            field_tok [f] += ntok

    val_loss = tot_loss / max(tot_tok, 1)
    tok_acc  = sum(field_corr) / max(tot_tok, 1)

    field_acc = {
        f"{name}_acc": (c / t if t else 0.0)
        for name, c, t in zip(field_names, field_corr, field_tok)
    }

    wandb.log({
        "val_loss":           val_loss,
        "val_token_accuracy": tok_acc,
        **field_acc
    }, commit=False)

    model.train()
    return {"val_loss": val_loss, "token_acc": tok_acc, **field_acc}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    signal.signal(signal.SIGINT, graceful_exit)

    # ---------- CLI ----------
    ap = argparse.ArgumentParser("Pre‑train TransactionModel (MLM)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config",   type=str, default="configs/pretrain.yaml")
    ap.add_argument("--data_dir", type=str, default="data")
    # common overrides
    ap.add_argument("--total_epochs", type=int)
    ap.add_argument("--batch_size",   type=int)
    ap.add_argument("--lr",           type=float)
    ap.add_argument("--build_windows", action="store_true",
                help="create the tensor file and exit")
    args_cli = ap.parse_args()

    # ---------- merge YAML + CLI ----------
    file_cfg = load_cfg(args_cli.config)
    args     = merge(args_cli, file_cfg)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- data ----------
    cache = Path(args.data_dir) / "bert_legit_processed.pt"
    if not cache.exists():
        raise FileNotFoundError("Run preprocessing.py first.")
    (train_df, val_df, _,
     enc, cat_feats, cont_feats, _, bin_edges) = torch.load(cache, weights_only=False)

    # ------------------------------------------------------------------
    # choose two cache paths (one per split)
    train_cache = Path(args.data_dir) / "processed_windows_train.pt"
    val_cache   = Path(args.data_dir) / "processed_windows_val.pt"

    # ------------------------------------------------------------------
    # training dataset  (build once if --build_windows is set)
    train_ds = get_dataset(
        train_df,
        cache_path = train_cache,
        build      = args.build_windows,
        group_by        = cat_feats[0],
        cat_features    = cat_feats,
        cont_features   = cont_feats,
        bin_edges       = bin_edges,
        window          = args.window,
        stride          = args.stride,
    )

    # ------------------------------------------------------------------
    # validation dataset  (same logic, separate file)
    val_ds = get_dataset(
        val_df,
        cache_path = val_cache,
        build      = args.build_windows,   # the same flag works for both
        group_by        = cat_feats[0],
        cat_features    = cat_feats,
        cont_features   = cont_feats,
        bin_edges       = bin_edges,
        window          = args.window,
        stride          = args.stride,
    )

    field_names = cat_feats + ["Amount"]
    train_loader = DataLoader(train_ds,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=collate_mlm,
                            num_workers=4,    # now safe on Windows
                            pin_memory=True,
                            persistent_workers=True,        
                            prefetch_factor=4)
    print(">>> dataset class:", type(train_loader.dataset).__name__)
    if hasattr(train_loader.dataset, "cat"):
        print(">>> tensors shape:", train_loader.dataset.cat.shape)
    else:
        print(">>> sliding‑window fallback!")
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=True,
                            collate_fn=collate_mlm,
                            num_workers=4,    # now safe on Windows
                            pin_memory=True,
                            persistent_workers=True,        
                            prefetch_factor=4)

    vocab_sizes = [len(enc[c]["inv"]) for c in cat_feats]
    bin_sizes   = [len(bin_edges[f]) + 2 for f in cont_feats]
    field_sizes = vocab_sizes + bin_sizes

    # ---------- model ----------
    ckpt_path = Path(args.data_dir) / "bert_backbone.pt"
    if args.resume and ckpt_path.exists():
        model, best_val, start_ep = load_ckpt(ckpt_path, device=device)
        cfg = model.cfg
    else:
        start_ep, best_val = 0, float("inf")
        cfg = ModelConfig(
            cat_vocab_sizes={k: len(enc[k]["inv"]) for k in cat_feats},
            cont_features  =cont_feats,
            emb_dropout    =args.emb_dropout,
            total_epochs   =args.total_epochs,
            window=args.window, stride=args.stride,
            ft_config = TransformerConfig(
                d_model=args.ft_d_model, n_heads=args.ft_n_heads,
                depth=args.ft_depth, ffn_mult=args.ft_ffn_mult,
                dropout=args.ft_dropout, layer_norm_eps=args.ft_layer_norm_eps,
                norm_first=args.ft_norm_first),
            seq_config = TransformerConfig(
                d_model=args.seq_d_model, n_heads=args.seq_n_heads,
                depth=args.seq_depth, ffn_mult=args.seq_ffn_mult,
                dropout=args.seq_dropout, layer_norm_eps=args.seq_layer_norm_eps,
                norm_first=args.seq_norm_first),
            lstm_config=None)
        model = TransactionModel(cfg).to(device)

    optim  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    # ---------- W&B ----------
    run = wandb.init(project="txn-transformer",
                     name=f"bert-pretrain-{Path(args.data_dir).stem}",
                     config=vars(args),
                     resume="allow" if args.resume else False)
    wandb.watch(model, log="all", log_freq=100)

    # ---------- training loop ----------
    patience, no_improve = 2, 0
    crit_cat = nn.CrossEntropyLoss(reduction="sum",  # for categorical fields
                               label_smoothing=0.1)
    bar_fmt = ("{l_bar}{bar:25}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) | "
               "elapsed: {elapsed} | ETA: {remaining}")
    eps_num   = 0.1          # same ϵ as the paper
    win       = 5            # ±5‑bin neighbourhood
    num_start = len(cat_feats)   # first index that belongs to a numeric field

    for ep in range(start_ep, args.total_epochs):
        model.train()
        tot_loss, tot_tok = 0.0, 0
        t0 = time.perf_counter()

        prog = tqdm(train_loader, mininterval=0.5, ncols=200, total=len(train_loader),
                    bar_format=bar_fmt,
                    desc=f"Epoch {ep+1}/{args.total_epochs}")
        batch_idx = 0
        for batch in prog:
            # ---------------- stack + GPU ----------------
            cat   = batch["cat"].to(device, non_blocking=True)     # (B,T,Fc)
            cont  = batch["cont"].to(device, non_blocking=True)    # (B,T,Fn_float)
            cbin  = batch["contbin"].to(device, non_blocking=True) # (B,T,Fn_bin)

            B, T, Fc = cat.shape; Fn = cbin.shape[2]; IGN = -100

            # ----------- random masks (GPU) -----------
            mcat = (torch.rand((B, T, Fc), device=device) < 0.15) | \
                (torch.rand((B, T, 1),  device=device) < 0.10)
            cat_labels = cat.masked_fill(~mcat, IGN)
            cat        = cat.masked_fill(mcat, 0)

            mnum = (torch.rand((B, T, Fn), device=device) < 0.15) | \
                (torch.rand((B, T, 1),  device=device) < 0.10)
            cont_labels = cbin.masked_fill(~mnum, IGN)
            cbin        = cbin.masked_fill(mnum, 0)

            labels_all = torch.cat((cat_labels, cont_labels), dim=-1)  # (B,T,F)
            mask_all   = labels_all != IGN                             # (B,T,F)
            F_tot = labels_all.size(-1)
            # ----------- forward (encode only once) -----------
            seq_flat  = model.encode(cat, cont)
            flat_lab  = labels_all.view(-1, F_tot)
            flat_mask = mask_all .view(-1, F_tot)

            loss_batch = torch.tensor(0.0, device=device)
            tok_batch  = 0

            with torch.amp.autocast("cuda"):
                for f, V in enumerate(field_sizes):
                    rows_f = flat_mask[:, f]
                    if not rows_f.any():
                        continue

                    logits_f = model.mlm_head_layers[f](seq_flat[rows_f])
                    labels_f = flat_lab[rows_f, f]

                    if f < num_start:                 # categorical
                        loss_batch += crit_cat(logits_f, labels_f)
                    else:                             # numeric
                        N = labels_f.size(0)
                        soft = torch.zeros((N, V), device=device, dtype=logits_f.dtype)
                        idx  = labels_f.unsqueeze(1)

                        # build neighbour‑smoothed distribution
                        soft.zero_()
                        p_val = eps_num / (2 * win)
                        offsets = torch.arange(-win, win + 1, device=device)
                        neigh   = torch.clamp(idx + offsets, 0, V - 1)
                        soft.scatter_add_(1, neigh, torch.full_like(neigh, p_val, dtype=soft.dtype))
                        soft.scatter_(1, idx, 1 - eps_num)

                        loss_batch += -(soft * F.log_softmax(logits_f, 1)).sum()

                    tok_batch += labels_f.numel()

            # ----- optimise -----
            scaler.scale(loss_batch / tok_batch).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            # ----- running totals -----
            tot_loss += loss_batch.item()
            tot_tok  += tok_batch

            prog.set_postfix({"loss": f"{tot_loss / tot_tok:.4f}"})
            batch_idx += 1
            if batch_idx >= 100:
                break


        train_loss = tot_loss / tot_tok
        val_stats  = evaluate_mlm(model, val_loader, field_sizes, field_names, device)
        val_loss   = val_stats["val_loss"]
        epoch_min  = (time.perf_counter() - t0) / 60.0

        wandb.log({"train_loss": train_loss,
                   "val_loss":   val_loss,
                   "epoch_min":  epoch_min})

        print(f"Epoch {ep+1:02}/{args.total_epochs} | "
              f"train {train_loss:.4f}  val {val_loss:.4f}  "
              f"time {epoch_min:.2f} min")

        if val_loss < best_val - 1e-5:
            best_val, no_improve = val_loss, 0
            save_ckpt(model, optim, ep, best_val, ckpt_path,
                      cat_feats, cont_feats, cfg)
            wandb.log({"best_val_loss": best_val})
            print("  ✔ new best — checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No val improvement for {patience} epochs ➜ early stop.")
                break

    # ---------- finish ----------
    art = wandb.Artifact("bert_backbone", type="model")
    art.add_file(str(ckpt_path));  wandb.log_artifact(art)
    run.finish();  print("Pre‑training complete.  Backbone saved.")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
