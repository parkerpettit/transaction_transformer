#!/usr/bin/env python
"""
Fine-tune TransactionModel for binary fraud detection.
Loads encoder weights from pretrained_backbone.pt and freezes them unless
--unfreeze is passed.
"""
from datetime import datetime
import argparse, time, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from config            import (LSTMConfig, MLPConfig)
from data.dataset      import TxnDataset, collate_fn
from evaluate   import evaluate_binary      # loss + acc per class
from utils             import save_ckpt, load_ckpt
import time
import sys
import traceback
from pathlib import Path
import torch
import torch.nn as nn
import wandb
from tqdm.auto import tqdm
from utils import load_cfg, merge, load_ckpt, resume_finetune
from model import FraudHeadMLP, LSTMHead
from torch.utils.data import DataLoader
from data.dataset import EmbeddingMemmapDataset
from torch.utils.data import WeightedRandomSampler
def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","1"):   return True
    if v in ("no","false","f","0"):   return False
    raise argparse.ArgumentTypeError("Boolean value expected")
def main():
    # --- 1. include --config flag -----------
    ap = argparse.ArgumentParser(description="Train / fine-tune TransactionModel")

    # ------------- paths / run control -------------------------------------------
    ap.add_argument("--resume",        action="store_true", help="Resume from latest checkpoint in data_dir")
    ap.add_argument("--config",        type=str,            help="YAML file with default hyper-params", default="configs/finetune.yaml")
    ap.add_argument("--data_dir",      type=str,            help="Root directory of raw or processed data")

    # ------------- training loop hyper-params ------------------------------------
    ap.add_argument("--total_epochs",  type=int,            help="Number of training epochs")
    ap.add_argument("--batch_size",    type=int,            help="Batch size for training")
    ap.add_argument("--lr",            type=float,          help="Initial learning rate")
    ap.add_argument("--window",        type=int,            help="Sequence length (transactions per sample)")
    ap.add_argument("--stride",        type=int,            help="Stride length between windows")

    # ------------- finetuning control -------------------------------------------
    ap.add_argument("--unfreeze",      action="store_true", help="Unfreeze the transformer backbone")

    # ------------- feature lists ------------------------------------------------
    ap.add_argument("--cat_features",  type=str,            help="Categorical column names (override YAML)", nargs="+")
    ap.add_argument("--cont_features", type=str,            help="Continuous column names  (override YAML)", nargs="+")

    # ------------- LSTM head -----------------------------------------------------
    ap.add_argument("--lstm_hidden",   type=int,            help="LSTM hidden size")
    ap.add_argument("--lstm_layers",   type=int,            help="Number of LSTM layers")
    ap.add_argument("--lstm_classes",  type=int,            help="Number of LSTM classes")
    ap.add_argument("--lstm_dropout",  type=float,          help="Dropout within LSTM")

    # ------------- MLP head ------------------------------------------------------
    ap.add_argument("--mlp_hidden_size", type=int, help="MLP hidden layer size")
    ap.add_argument("--mlp_num_layers",  type=int, help="Number of MLP layers")
    ap.add_argument("--mlp_dropout",     type=float, help="MLP dropout")

    ap.add_argument(
        "--head",
        choices=["mlp", "lstm"],
        default="mlp",
        help="Classification head to attach on top of the frozen backbone",
    )

    cli = ap.parse_args()



    # --- 2. merge file + CLI ---------------
    file_params = load_cfg(cli.config)
    args = merge(cli, file_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data -------------------------------------------------------------------
    # cache = Path(args.data_dir) / "full_processed.pt"
    # if cache.exists():
    #     print("Processed data exists, loading now.")
    #     train_df, val_df, test_df, enc, cat_features, cont_features, scaler = torch.load(cache,  weights_only=False)
    #     print("Processed data loaded.")
    # else:
    #     print("Preprocessed data not found.")
    #     raw = Path(args.data_dir) / "card_transaction.v1.csv"
    #     train_df, val_df, test_df, enc, cat_features, cont_features, scaler = preprocess(raw, args.cat_features, args.cont_features)
    #     print("Finished processing data. Now saving.")
    #     torch.save((train_df, val_df, test_df, enc, cat_features, cont_features, scaler), cache)
    #     print("Processed data saved.")

    # pretrain_train_ds = TxnDataset(train_df, cat_features[0], cat_features, cont_features,
    #                args.window, args.stride)
    # pretrain_val_ds = TxnDataset(val_df, cat_features[0], cat_features, cont_features,
    #                args.window, args.stride)






    # progress bar format (reuse from pretrain)
    bar_fmt = (
        "{l_bar}{bar:25}| "
        "{n_fmt}/{total_fmt} batches "
        "({percentage:3.0f}%) | "
        "elapsed: {elapsed} | ETA: {remaining} | "
        "{rate_fmt} | "
        "{postfix}"
    )

    ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = Path(args.data_dir) / f"big_finetune_{args.head}.ckpt"
    run_name  = f"finetune-{args.head}-{Path(args.data_dir).stem}-{ts}"
    tag       = f"{args.head}_{'unfrozen' if args.unfreeze else 'frozen'}"
    backbone_path = Path(args.data_dir) / "big_legit_backbone.pt"


    # print("Loading train_data")
    # # train_ds = EmbeddingMemmapDataset("data/train_embedded_data")


    # print("Loading val_data")
    # val_ds = EmbeddingMemmapDataset("data/val_embedded_data")


        # --- Data -------------------------------------------------------------------
    cache = Path(args.data_dir) / "full_processed.pt"
    if cache.exists():
        print("Processed data exists, loading now.")
        train_df, val_df, test_df, enc, cat_features, cont_features, scaler = torch.load(cache,  weights_only=False)
        print("Processed data loaded.")
    else:
        print("Preprocessed data not found.")
        
    train_ds = TxnDataset(train_df, cat_features[0], cat_features, cont_features,
               args.window, args.stride)
    val_ds = TxnDataset(val_df, cat_features[0], cat_features, cont_features,
               args.window, args.stride)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size = args.batch_size,
        collate_fn = collate_fn,
        num_workers=4
        )

    val_loader = DataLoader(
        val_ds,
        batch_size = args.batch_size,
        shuffle=False,
        collate_fn = collate_fn,
        num_workers=4
        )

    if args.resume and ckpt_path.exists():
        # resume the entire fine‑tune run
        model, best_val, start_epoch, optim = resume_finetune(
            ckpt_path,
            unfreeze_backbone=args.unfreeze,
            device=device,
        )
        print(f"Resuming fine-tune from epoch {start_epoch}, best val={best_val:.4f}")
        forward_mode = args.head

    else:  # fresh fine‑tune from pretrained backbone
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found at {backbone_path}")

        # 1) load frozen backbone (weights + cfg)
        model, _, _, = load_ckpt(backbone_path, device=device)
        cfg = model.cfg
        print(f"Loaded backbone from {backbone_path}")

        # after loading the backbone
        if args.head == "mlp":
            cfg.mlp_config = MLPConfig(
                hidden_size = args.mlp_hidden_size,
                num_layers  = args.mlp_num_layers,
                dropout     = args.mlp_dropout,
            )
            model.add_mlp_head(cfg.mlp_config)
            forward_mode = "mlp"

        elif args.head == "lstm":
            cfg.lstm_config = LSTMConfig(
                hidden_size = args.lstm_hidden,
                num_layers  = args.lstm_layers,
                num_classes = args.lstm_classes,
                dropout     = args.lstm_dropout,
            )
            model.add_lstm_head(cfg.lstm_config)
            forward_mode = "lstm"


        # 3) freeze or unfreeze backbone
        if not args.unfreeze:
            for n, p in model.named_parameters():
                if not n.startswith(f"{args.head}_head"):
                    p.requires_grad = False
        for n, p in model.named_parameters():
            print(n, p.requires_grad)
        # 4) optimiser on trainable params only
        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )
        start_epoch = 0
        best_val = float("inf")
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size = args.batch_size,
    #     shuffle=False,
    #     collate_fn = collate_fn,
    # )

    # if args.head == "mlp":
    #     cfg = MLPConfig(
    #         hidden_size = args.mlp_hidden_size,
    #         num_layers  = args.mlp_num_layers,
    #         dropout     = args.mlp_dropout,
    #     )
    #     model = FraudHeadMLP(emb_dim=train_ds.M, cfg=cfg)


    # elif args.head == "lstm":
    #     cfg = LSTMConfig(
    #         hidden_size = args.lstm_hidden,
    #         num_layers  = args.lstm_layers,
    #         num_classes = args.lstm_classes,
    #         dropout     = args.lstm_dropout,
    #     )
    #     model = LSTMHead(cfg=cfg, input_size=train_ds.M)
    # else:
    #     raise Exception("No classification head specified. Use --head mlp or --head lstm")


    # start_epoch = 0
    # best_val = float("inf")
    # model.to(device)


    # train_loader = DataLoader(
    #     train_ds,
    #     shuffle=True,
    #     batch_size = args.batch_size,
    #     num_workers=4
    # )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size = args.batch_size,
    #     shuffle=False,
    #     num_workers=4
    # )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cfg = model.cfg
    # --------------------------------------------------------------------------- #
    #  loss
    # --------------------------------------------------------------------------- #
    pos_weight_tensor = torch.tensor(82, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scheduler = CosineAnnealingLR(
    optim,
    T_max=args.total_epochs,   # number of epochs to anneal over
    eta_min=1e-7,              # final minimum LR
    )   

    print("Starting fine-tune training loop")
    best_f1 = -float('inf')
    if start_epoch >= args.total_epochs:
        raise IndexError(
            f"Start epoch ({start_epoch}) >= total_epochs ({args.total_epochs})"
        )

    # Initialize Weights & Biases
    run = wandb.init(
        project="finetune",
        name   = run_name,
        config = vars(args),
        resume = "auto",
        tags = [tag]
    )
    wandb.watch(model, log="parameters", log_freq=1000)

    # Early-stopping setup
    patience = 5
    ep_no_improve = 0
    model.to(device)
    try:
        for ep in range(start_epoch, args.total_epochs):
            prog_bar = tqdm(
                train_loader,
                desc=f"Epoch {ep+1}/{args.total_epochs}",
                unit="batch",
                total=len(train_loader),
                bar_format=bar_fmt,
                ncols=200,
                leave=True,
            )

            model.train()
            tot_loss = 0.0
            sample_count = 0
            t0 = time.perf_counter()
            batch_idx = 0
            # for batch in prog_bar:
            #     delta = batch["actual_embedding"].to(device)
                
            #     label = batch["label"].to(device)
            #     logits = model(delta)  # (B)

            #     loss = criterion(logits, label.float())
            #     if batch_idx % 100 == 0:
            #         wandb.log({"training_loss": loss.item()})

            #     optim.zero_grad()
            #     loss.backward()
            #     optim.step()
            #     scheduler.step()

            #     batch_size = label.size(0)
            #     tot_loss += loss.item() * batch_size
            #     sample_count += batch_size
            #     prog_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            #     batch_idx += 1

            # train_loss = tot_loss / sample_count
            # # -- validation ------------------------------------------------------
            # val_loss, val_metrics = evaluate_binary(
            #     model, val_loader, criterion, device,
            #     class_names=["non-fraud", "fraud"],
            # )
            for batch in prog_bar:
                cat_inp, cont_inp, labels = batch["cat"].to(device), batch["cont"].to(device), batch['label'].to(device)
                labels = labels.float()
                logits = model(cat_inp, cont_inp, mode=forward_mode)  # (B)

                loss = criterion(logits, labels)
                if batch_idx % 100 == 0:
                    wandb.log({"training_loss": loss.item()})
                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()

                batch_size = labels.size(0)
                tot_loss += loss.item() * batch_size
                sample_count += batch_size
                prog_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                batch_idx += 1

            train_loss = tot_loss / sample_count
            # -- validation ------------------------------------------------------
            val_loss, val_metrics = evaluate_binary(
                model, val_loader, criterion, device,
                class_names=["non-fraud", "fraud"], mode=forward_mode
            )
            epoch_time_min = (time.perf_counter() - t0) / 60.0
            wandb.log({"epoch_time_min": epoch_time_min})

            print(
                f"Epoch {ep+1}/{args.total_epochs} | "
                f"train {train_loss:.4f} | "
                f"val {val_loss:.4f} "
                f"(acc {val_metrics['accuracy']*100:.2f}%, "
                f"F1 {val_metrics['f1']:.3f}, "
                f"AUC {val_metrics['roc_auc']:.3f}) | "
                f"{epoch_time_min:.2f} min"
            )
            # -- early-stopping / checkpoint -------------------------------------
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                ep_no_improve = 0
                print("New best validation loss. Saving checkpoint.")
                save_ckpt(
                    model, optim, ep, best_val,
                    ckpt_path, args.cat_features, args.cont_features, cfg
                )
                # type: ignore
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics["f1"]
                    print("New best f1 score. Saving checkpoint.")
            elif val_metrics['f1'] > best_f1:
                ep_no_improve = 0
                best_f1 = val_metrics["f1"]
                print("New best f1 score Saving checkpoint.")
                save_ckpt(
                    model, optim, ep, best_val,
                    ckpt_path, args.cat_features, args.cont_features, cfg
                )
            else:
                ep_no_improve += 1
                if ep_no_improve >= patience:
                    print(f"No improvement for {patience} epochs. Stopping early.")
                    break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            run.alert(
                title="CUDA OOM",
                text=f"OOM at batch_size={args.batch_size}. Marking failed."
            )
            run.finish(exit_code=97)
            sys.exit(97)
        else:
            traceback.print_exc()
            run.finish(exit_code=98)
            sys.exit(98)

    finally:
        if wandb.run is not None and ckpt_path.exists():
            artifact = wandb.Artifact("finetune-model", type="model")
            artifact.add_file(str(ckpt_path))
            wandb.log_artifact(artifact)
            wandb.log({"best_val_loss": best_val,
                    "best_f1": best_f1})
            run.finish()
        print(f"Fine-tuning complete. Best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()