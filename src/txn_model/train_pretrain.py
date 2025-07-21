#!/usr/bin/env python
"""
Pre-train TransactionModel to predict the next transaction
(categorical fields + continuous amount).  No fraud head.
Saves encoder weights to  pretrained_backbone.pt
"""
from cProfile import label
import wandb
import argparse, time, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from config            import (ModelConfig, TransformerConfig)
from data.dataset      import TxnDataset, collate_fn, slice_batch
# from data.preprocessing import preprocess
from model import TransactionModel
from evaluate          import evaluate            # per-feature val metrics

import yaml
from utils import load_cfg, merge, load_ckpt
import time
from tqdm.auto import tqdm  
from utils import save_ckpt
import signal, sys
import wandb, torch, traceback, sys



def main():
    def show_samples(cat_inp, cat_tgt, cat_preds,
                    cont_tgt, cont_preds,
                    enc, cat_features, cont_features,
                    n=3):
        """
        cat_inp   : [B, W, F_cat]  (not used here but available if you want)
        cat_tgt   : [B, F_cat]     codes of last timestep
        cat_preds : [B, F_cat]     argmax codes
        cont_tgt  : [B, F_cont]
        cont_preds: [B, F_cont]
        """
        B = cat_tgt.size(0)
        n = min(n, B)

        # decode function
        def d(code, feat_name):
            inv = enc[feat_name]["inv"]
            return inv[code] if code < len(inv) else f"<UNK:{code}>"

        for b in range(n):
            print(f"\n─ Sample {b} ─")
            for i, feat in enumerate(cat_features):
                tgt_code  = cat_tgt[b, i].item()
                pred_code = cat_preds[b, i].item()
                print(f"{feat:<18}: tgt={d(tgt_code, feat)} | pred={d(pred_code, feat)}")
            for j, feat in enumerate(cont_features):
                t_val = cont_tgt[b, j].item()
                p_val = cont_preds[b, j].item()
                print(f"{feat:<18}: tgt={t_val:>8.2f} | pred={p_val:>8.2f}")
            print("─" * 40)

    def graceful_exit(signum=None, frame=None):
        """Save state, finish wandb, and exit immediately."""
        # # 1. Try to save a last-minute checkpoint (optional but useful)
        # if 'model' in globals():
        #     ckpt_int = Path(args.data_dir) / "pretrained_backbone_interrupt.pt"
        #     torch.save(model.state_dict(), ckpt_int)
        #     print(f"\n[Ctrl-C] Saved interrupt checkpoint → {ckpt_int}")

        print("[Ctrl-C] Exiting now.")
        sys.exit(0)

    # Register SIGINT (Ctrl-C) handler *before* anything long-running starts
    signal.signal(signal.SIGINT, graceful_exit)



    ap = argparse.ArgumentParser(description="Train / fine-tune TransactionModel")


    # ───────────── paths / run control ───────────────────────────────────────────
    ap.add_argument("--resume", action="store_true",    help="Resume from latest checkpoint in data_dir")
    ap.add_argument("--config",            type=str,    help="YAML file with default hyper-params", default="configs/pretrain.yaml")
    ap.add_argument("--data_dir",          type=str,    help="Root directory of raw or processed data")

    # ───────────── training loop hyper-params ────────────────────────────────────
    ap.add_argument("--total_epochs",      type=int,    help="Number of training epochs")
    ap.add_argument("--batch_size",        type=int,    help="Batch size for training")
    ap.add_argument("--lr",                type=float,  help="Initial learning rate")
    ap.add_argument("--window",            type=int,    help="Sequence length (transactions per sample)")
    ap.add_argument("--stride",            type=int,    help="Stride length between windows")

    # ───────────── feature lists ────────────────────────────────────────────────
    ap.add_argument("--cat_features",      type=str,    help="Categorical column names (override YAML)", nargs="+")
    ap.add_argument("--cont_features",     type=str,    help="Continuous column names  (override YAML)", nargs="+")

    # ───────────── architecture: embedding layer ────────────────────────────────
    ap.add_argument("--emb_dropout",        type=float, help="Dropout after embedding layer")

    # ── Field-level transformer (intra-row) ──────────────────────────────────────
    ap.add_argument("--ft_d_model",         type=int,   help="Field-transformer hidden dimension")
    ap.add_argument("--ft_depth",           type=int,   help="Field-transformer number of layers")
    ap.add_argument("--ft_n_heads",         type=int,   help="Field-transformer number of attention heads")
    ap.add_argument("--ft_ffn_mult",        type=int,   help="Field-transformer feedforward expansion factor")
    ap.add_argument("--ft_dropout",         type=float, help="Dropout within field-transformer")
    ap.add_argument("--ft_layer_norm_eps",  type=float, help="Layer norm epsilon for field-transformer")
    ap.add_argument("--ft_norm_first",      type=bool,  help="Norm first for field-transformer")
    # ── Sequence-level transformer (inter-row) ───────────────────────────────────
    ap.add_argument("--seq_d_model",        type=int,   help="Sequence-transformer hidden dimension")
    ap.add_argument("--seq_depth",          type=int,   help="Sequence-transformer number of layers")
    ap.add_argument("--seq_n_heads",        type=int,   help="Sequence-transformer number of attention heads")
    ap.add_argument("--seq_ffn_mult",       type=int,   help="Sequence-transformer feedforward expansion factor")
    ap.add_argument("--seq_dropout",        type=float, help="Dropout within sequence-transformer")
    ap.add_argument("--seq_layer_norm_eps", type=float, help="Layer norm epsilon for sequence-transformer")
    ap.add_argument("--seq_norm_first",     type=bool,  help="Norm first for sequence-transformer")


    # ── Final classification layer ──────────────────────────────────────────────
    ap.add_argument("--clf_dropout",  type=float, help="Dropout before final classification layer")

    cli = ap.parse_args()



    # --- 2. merge file + CLI ---------------
    file_params = load_cfg(cli.config)
    args = merge(cli, file_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # ─── Data (load cache or create) ───────────────────────────────────────────

    cache = Path(args.data_dir) / "legit_processed.pt"
    if cache.exists():
        print("Processed data exists, loading now.")
        train_df, val_df, test_df, enc, cat_features, cont_features, scaler = torch.load(cache,  weights_only=False)
        print("Processed data loaded.")
    else:
        print("Processed data file doesn't exist, run preprocessing.py.")
        # print("Preprocessed data not found. Processing now.")
        # raw = Path(args.data_dir) / "card_transaction.v1.csv"
        # train_df, val_df, test_df, enc, cat_features, cont_features, scaler = preprocess(raw, args.cat_features, args.cont_features)
        # print("Finished processing data. Now saving.")
        # torch.save((train_df, val_df, test_df, enc, cat_features, cont_features, scaler), cache)
        # print("Processed data saved.")
    print("Creating training loader")
    train_loader = DataLoader(
        TxnDataset(train_df, cat_features[0], cat_features, cont_features,
                args.window, args.stride),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print("Creating validation loader")
    val_loader   = DataLoader(
        TxnDataset(val_df, cat_features[0], cat_features, cont_features,
                args.window, args.stride),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)


    print("Starting training loop")

    bar_fmt = (
        "{l_bar}{bar:25}| "         # visual bar
        "{n_fmt}/{total_fmt} batches "  # absolute progress
        "({percentage:3.0f}%) | "   # %
        "elapsed: {elapsed} | ETA: {remaining} | "  # timing
        "{rate_fmt} | "             # batches / sec
        "{postfix}"                 # losses go here
    )


    ckpt_path = (Path(args.data_dir) / "big_legit_backbone.pt")

    if args.resume:
        model, best_val, start_epoch = load_ckpt(ckpt_path, device=device)
        model.to(device)
        cfg = model.cfg
    else:

        cfg = ModelConfig(
            cat_vocab_sizes = {k: len(enc[k]["inv"]) for k in cat_features}, # dont take from args because preprocessing may change it
            cont_features   = cont_features, # same as above
            emb_dropout     = args.emb_dropout,
            padding_idx     = 0,
            total_epochs    = args.total_epochs,
            window          = args.window,
            stride          = args.stride,


            ft_config = TransformerConfig(
                d_model        = args.ft_d_model,         
                n_heads        = args.ft_n_heads,  
                depth          = args.ft_depth,   
                ffn_mult       = args.ft_ffn_mult, 
                dropout        = args.ft_dropout, 
                layer_norm_eps = args.ft_layer_norm_eps,
                norm_first     = args.ft_norm_first,
            ),

            seq_config = TransformerConfig(
                d_model        = args.seq_d_model,
                n_heads        = args.seq_n_heads,
                depth          = args.seq_depth,
                ffn_mult       = args.seq_ffn_mult,
                dropout        = args.seq_dropout,
                layer_norm_eps = args.seq_layer_norm_eps,
                norm_first     = args.seq_norm_first,
            ),

            lstm_config = None,   # pre-train phase keeps this off
        )
        print("Initializing model")
        model = TransactionModel(cfg).to(device)

        
        start_epoch = 0
        best_val  = float("inf")

    print("new train")
    vocab_sizes = list(cfg.cat_vocab_sizes.values()) # type: ignore
    crit_cat  = nn.CrossEntropyLoss(label_smoothing=0.1)
    crit_cont = nn.MSELoss()
    optim     = torch.optim.Adam(model.parameters(), lr=args.lr)
    if start_epoch >= args.total_epochs:
        raise IndexError("Start epoch from loaded model is greater than total epochs.")
        
    # ─── wandb must know we’re resuming ───────────────────────────────────────
    run = wandb.init(
        project="txn-transformer",
        name   = f"pretrain-{Path(args.data_dir).stem}",
        config = vars(args),
        resume = "allow" if args.resume else False,    
    )

    wandb.watch(model, log="all", log_freq=100)

    # ─── Training loop ─────────────────────────────────────────────────────────
    patience = 2 # number of acceptable consecutive epochs without validation loss improvement
    ep_without_improvement = 0
    # wandb.log({"batches_per_epoch": len(train_loader)})
    try:
        for ep in range(start_epoch, args.total_epochs):
            prog_bar = tqdm(
                train_loader,
                desc=f"Epoch {ep+1}/{args.total_epochs}",
                unit="batch",
                total=len(train_loader),
                bar_format=bar_fmt,
                ncols=200,               # wider for readability 
                leave=True,       
            )
            
            model.train()
            tot_loss = 0
            epoch_sample_count = 0
            t0 = time.perf_counter()
            batch_idx = 0

            for batch in prog_bar:
                cat_input, cont_inp, cat_tgt, cont_tgt, _ = (t.to(device) for t in slice_batch(batch))
                cat_logits, cont_pred = model(cat_input, cont_inp, mode="ar")

                # categorical losses field-wise
                start = 0
                loss_cat = 0
                for i, vocab_len in enumerate(vocab_sizes):
                    loss_cat += crit_cat(cat_logits[:, start:start+vocab_len], cat_tgt[:, i])
                    start += vocab_len
                loss_cont = crit_cont(cont_pred, cont_tgt)
                # loss_cat /= len(vocab_sizes) # average cat loss across number of categories
                # only one continuous feature, give each head equal weight
                loss = (loss_cat + loss_cont) / (len(vocab_sizes) + 1 )
                wandb.log({"training_loss": loss.item()})
                optim.zero_grad()
                loss.backward()
                optim.step()
                batch_size = cat_input.size(0)
                tot_loss += loss.item() * batch_size
                epoch_sample_count += batch_size

                prog_bar.set_postfix({
                    "tot":  f"{loss.item():.4f}",
                    "cat":  f"{loss_cat.item():.4f}", # type: ignore
                    "cont": f"{loss_cont.item():.4f}",
                })
            train_loss = tot_loss / epoch_sample_count
           

            val_loss, feat_acc = evaluate(model, val_loader, cat_features,
                                        {f: len(enc[f]["inv"]) for f in cat_features},
                                        crit_cat, crit_cont, device)
            t1 = time.perf_counter()
            time_elapsed = (t1 - t0) / 60
            wandb.log({
            " val_loss":  val_loss,
            **{f"validation_accuracy_{k}": v for k, v in feat_acc.items()}
            }, commit=False)
            wandb.log({
            "epoch_time_min": time_elapsed,
            }, commit=True)
            print(f"Epoch {ep+1:02}/{args.total_epochs} "
                f"| train {train_loss:.4f}  val {val_loss:.4f} "
                f"| Time to complete epoch: {time_elapsed:2f} minutes")
            
            if val_loss < best_val - 1e-5:  
                ep_without_improvement = 0
                print("New validation loss better than previous. Saving checkpoint.")             
                best_val = val_loss                       
                ckpt_path = Path(args.data_dir) / "big_legit_backbone.pt"
                save_ckpt(                               
                    model, optim, ep, best_val,
                    ckpt_path, cat_features, cont_features, cfg # type: ignore
                )
                wandb.log({"best_val_loss": best_val}) 
                print(f"New best ({best_val:.4f}), checkpoint saved.")
            else:
                if ep_without_improvement >= patience:
                    print(f"No improvement for {patience} epochs. Stopping early.")
                    break
                else:
                    ep_without_improvement += 1
            # after computing preds for this batch
            if batch_idx == 0:   # print only from the first batch to avoid spam
                # gather predictions in the same per-feature layout used for accuracy
                start = 0
                preds_cat = torch.empty_like(cat_tgt)
                for i, V in enumerate(vocab_sizes):
                    end = start + V
                    preds_cat[:, i] = cat_logits[:, start:end].argmax(dim=1)
                    start = end

                show_samples(
                    cat_input, cat_tgt, preds_cat,
                    cont_tgt,  cont_pred,
                    enc, cat_features, cont_features,
                    n=1
                )
            batch_idx += 1


                    
    except RuntimeError as e:
            # graceful handling of CUDA OOM
            if "out of memory" in str(e).lower():
                run.alert(
                    title="CUDA OOM",
                    text=f"Run crashed at batch={args.batch_size}.  Marking failed.")
                run.finish(exit_code=97)      # any non-zero exit marks failure
                sys.exit(97)
            else:
                traceback.print_exc()
                run.finish(exit_code=98)
                sys.exit(98)

    return args, run




if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    args, run = main()
    ckpt_path = Path(args.data_dir) / "big_legit_backbone.pt"
    if wandb.run is not None:
        artifact = wandb.Artifact("big_legit_backbone", type="model")
        artifact.add_file(str(ckpt_path))
        wandb.log_artifact(artifact)
        run.finish()   
    print("Pre-training complete.  Backbone saved.")
