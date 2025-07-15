import os
import argparse
import numpy as np
import torch
import lightgbm as lgb
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# -----------------------------------
# Command-line arguments and W&B init
# -----------------------------------
parser = argparse.ArgumentParser(description="End-to-end training script for LightGBM with Transformer embeddings")
parser.add_argument("--data_dir", type=str, default="data/", help="Path to data directory (if needed by TxnDataset)")
parser.add_argument("--model_checkpoint", type=str, default="data/pretrained_backbone.pt", help="Path to the pretrained TransactionModel checkpoint file")
parser.add_argument("--wandb_project", type=str, default="lightgbm", help="W&B project name")
parser.add_argument("--wandb_run_name", type=str, default="test", help="W&B run name (optional)")
parser.add_argument("--feature_dir", type=str, default="features", help="Directory to save or load feature .npy files")
parser.add_argument("--rebuild", action="store_true", help="Re-extract embeddings even if .npy files exist")
parser.add_argument("--in_memory", action="store_true", help="Do not write features to disk; keep datasets in memory (requires sufficient RAM)")
# Hyperparameters for LightGBM (with defaults that can be overridden)
parser.add_argument("--learning_rate", type=float, default=0.1, help="LightGBM learning rate")
parser.add_argument("--num_leaves", type=int, default=10000, help="Number of leaves in LightGBM tree")
parser.add_argument("--max_depth", type=int, default=20, help="Max tree depth (<=0 means no limit)")
parser.add_argument("--n_estimators", type=int, default=1000, help="Number of boosting rounds (trees)")
parser.add_argument("--subsample", type=float, default=1.0, help="Subsample (bagging fraction)")
parser.add_argument("--subsample_freq", type=int, default=0, help="Bagging frequency (0 means no subsampling)")
parser.add_argument("--feature_fraction", type=float, default=1.0, help="Feature fraction (colsample_bytree)")
parser.add_argument("--min_data_in_leaf", type=int, default=1, help="Minimum data per leaf")
parser.add_argument("--min_sum_hessian_in_leaf", type=float, default=1e-6, help="Minimum sum of Hessian (leaf min_weight)")
parser.add_argument("--min_gain_to_split", type=float, default=0.0, help="Minimum gain to split a node")
parser.add_argument("--lambda_l1", type=float, default=0.0, help="L1 regularization")
parser.add_argument("--lambda_l2", type=float, default=0.0, help="L2 regularization")
parser.add_argument("--is_unbalance", action="store_true", help="Handle unbalanced classes by weighting positives automatically")
parser.add_argument("--scale_pos_weight", type=float, default=1.0, help="Weight for positive class (use if not using is_unbalance)")
parser.add_argument("--max_bin", type=int, default=255, help="Max bins for histogram binning")
parser.add_argument("--boosting_type", type=str, default="gbdt", choices=["gbdt","dart","goss"], help="Boosting type")
parser.add_argument("--objective", type=str, default="binary", help="Objective for LightGBM")
parser.add_argument("--metric", type=str, default="average_precision", help="Evaluation metric for LightGBM")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--verbosity", type=int, default=1, help="LightGBM verbosity (-1 to disable warnings)")
parser.add_argument("--early_stopping_rounds", type=int, default=500, help="Early stopping rounds")
parser.add_argument("--eval_log_interval", type=int, default=1, help="Interval (in rounds) for logging evaluation metrics")
args = parser.parse_args()

# Initialize W&B run (for hyperparameter tracking and logging)
# We pass all hyperparameters in wandb.config for easy sweep overrides
wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
config = wandb.config  # to simplify access

# -----------------------------------
# Load pretrained model and dataset
# -----------------------------------
# Import the TransactionModel and TxnDataset classes (assuming they're available in your environment)
from model import TransactionModel  # user-provided module
from data.dataset import TxnDataset, collate_fn       # user-provided module (or similar)
from utils import load_ckpt
from torch.utils.data import DataLoader
from pathlib import Path
# Load the pretrained transformer model (TransactionModel)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone, _, _ = load_ckpt(Path(config.model_checkpoint))
model = backbone.to(device)

# Move model to GPU if available for faster inference
model.eval()  # set model to evaluation mode
# ───── 1) Load or preprocess raw DataFrames ───────────────────────────
cache = Path(config.data_dir) / "processed_data.pt"

print("Loading processed_data.pt ...")
train_df, val_df, test_df, enc, cat_feats, cont_feats, scaler = torch.load(cache, weights_only=False)

# ───── 2) Build DataLoaders ────────────────────────────────────────────
features_dir = Path(config.data_dir) / "features"
features_dir.mkdir(parents=True, exist_ok=True)

train_loader = DataLoader(
    TxnDataset(train_df, cat_feats[0], cat_feats, cont_feats, 10, 5),
    batch_size=384, shuffle=True,
    collate_fn=collate_fn, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    TxnDataset(val_df, cat_feats[0], cat_feats, cont_feats, 10, 5),
    batch_size=384, shuffle=False,
    collate_fn=collate_fn, num_workers=0, pin_memory=True
)
test_loader = DataLoader(
    TxnDataset(test_df, cat_feats[0], cat_feats, cont_feats, 10, 5),
    batch_size=384, shuffle=False,
    collate_fn=collate_fn, num_workers=0, pin_memory=True
)



# Ensure the feature directory exists
os.makedirs(args.feature_dir, exist_ok=True)
feature_paths = { 
    "train_emb": os.path.join(args.feature_dir, "train_emb.npy"),
    "train_lbl": os.path.join(args.feature_dir, "train_lbl.npy"),
    "val_emb": os.path.join(args.feature_dir, "val_emb.npy"),
    "val_lbl": os.path.join(args.feature_dir, "val_lbl.npy"),
    "test_emb": os.path.join(args.feature_dir, "test_emb.npy"),
    "test_lbl": os.path.join(args.feature_dir, "test_lbl.npy")
}

# -----------------------------------
# Extract embeddings if missing
# -----------------------------------
need_extraction = args.rebuild or not all(os.path.isfile(p) for p in feature_paths.values())
if need_extraction:
    print("Extracting embeddings using the TransactionModel…")


    from numpy.lib.format import open_memmap

    def extract_to_array(loader, emb_path, lbl_path):
        n_samples = len(loader.dataset)
        # Peek one batch to infer embedding size
        example_batch = next(iter(loader))
        # Prepare inputs dict for the model
        example_inputs = {
            k: v.to(device)
            for k, v in example_batch.items()
            if k in ("cat", "cont", "pad_mask")
        }
        with torch.no_grad():
            example_out = model(**example_inputs, mode="lightgbm")
        # figure out embedding dim M
        if example_out.dim() == 3:
            M = example_out.size(-1)
        elif example_out.dim() == 2:
            M = example_out.size(1)
        else:
            raise RuntimeError(f"Unexpected model output shape {example_out.shape}")

        # Preallocate memmaps (or numpy arrays if in_memory)
        if args.in_memory:
            emb_array = np.empty((n_samples, M), dtype=np.float32)
            lbl_array = np.empty((n_samples,), dtype=np.int64)
        else:
            emb_array = open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n_samples, M))
            lbl_array = open_memmap(lbl_path, mode="w+", dtype=np.int64,   shape=(n_samples,))

        idx = 0
        with torch.no_grad():
            for batch in loader:
                # split inputs vs. label
                inputs = {k: batch[k].to(device) for k in ("cat", "cont", "pad_mask")}
                labels = batch["label"].to(device)

                out = model(**inputs, mode="lightgbm")
                # take last token if sequence output
                if out.dim() == 3:
                    emb = out[:, -1, :].cpu().numpy()
                else:
                    emb = out.cpu().numpy()

                lbl = labels.cpu().numpy()
                B = emb.shape[0]

                emb_array[idx : idx + B, :] = emb
                lbl_array[idx : idx + B]    = lbl
                idx += B

        # flush to disk if using memmap
        if not args.in_memory:
            emb_array.flush()
            lbl_array.flush()

    # Extract for each split
    extract_to_array(train_loader, feature_paths["train_emb"], feature_paths["train_lbl"])
    extract_to_array(val_loader,   feature_paths["val_emb"],   feature_paths["val_lbl"])
    extract_to_array(test_loader,  feature_paths["test_emb"],  feature_paths["test_lbl"])

    print("Done extracting embeddings.")



else:
    print("Embedding files found - skipping extraction (use --rebuild to force).")

# -----------------------------------
# Load embeddings for LightGBM
# -----------------------------------
if args.in_memory and need_extraction:
    # If just extracted in memory, we already have the arrays in memory from the extraction function
    # They would be returned or stored globally; to keep code simple, the extract_to_array function writes to outer scope variables.
    # For clarity, we reload from disk even if in_memory was used with rebuild, because we saved nothing in that mode.
    # Instead, we will treat in_memory+rebuild as meaning we have them in local scope via closure (not returned though).
    # As a workaround, if in_memory and extraction just happened, we reload from memory through an alternative path.
    pass

# If not in_memory, or if in_memory but files already existed (no extraction just done), load from files
X_train = None; y_train = None
X_val = None; y_val = None
X_test = None; y_test = None
if args.in_memory:
    # Load fully into memory
    X_train = np.load(feature_paths["train_emb"], mmap_mode=None) if os.path.isfile(feature_paths["train_emb"]) else None
    y_train = np.load(feature_paths["train_lbl"], mmap_mode=None) if os.path.isfile(feature_paths["train_lbl"]) else None
    X_val   = np.load(feature_paths["val_emb"], mmap_mode=None)   if os.path.isfile(feature_paths["val_emb"]) else None
    y_val   = np.load(feature_paths["val_lbl"], mmap_mode=None)   if os.path.isfile(feature_paths["val_lbl"]) else None
    X_test  = np.load(feature_paths["test_emb"], mmap_mode=None)  if os.path.isfile(feature_paths["test_emb"]) else None
    y_test  = np.load(feature_paths["test_lbl"],  mmap_mode=None)
else:
    # Memory-map the features (read-only mode) to avoid loading entire arrays into RAM
    X_train = np.load(feature_paths["train_emb"], mmap_mode="r")
    y_train = np.load(feature_paths["train_lbl"], mmap_mode="r")
    X_val   = np.load(feature_paths["val_emb"], mmap_mode="r")
    y_val   = np.load(feature_paths["val_lbl"], mmap_mode="r")
    X_test  = np.load(feature_paths["test_emb"], mmap_mode="r")
    y_test  = np.load(feature_paths["test_lbl"], mmap_mode="r")

# -----------------------------------
# Train LightGBM model
# -----------------------------------
# Create LightGBM Dataset objects

import numpy as np

# 1) Basic info
print("X_train shape:", X_train.shape)
print("y_train type:", type(y_train))
print("y_train shape:", getattr(y_train, "shape", None))
print("y_train dtype:", getattr(y_train, "dtype", None))

# 2) Backing file info (if y_train is memmap)
if hasattr(y_train, "filename"):
    print("y_train file:", y_train.filename)

# 3) Peek at the first few labels
print("First 10 y_train:", y_train[:10])

# 4) Check for NaNs or Infs
print("X_train has NaN?", np.isnan(X_train).any())
print("X_train has Inf?", np.isinf(X_train).any())
print("y_train has NaN?", np.isnan(y_train).any())
print("y_train has Inf?", np.isinf(y_train).any())

# 5) Check label distribution (for a binary objective)
vals, counts = np.unique(y_train, return_counts=True)
print("y_train unique values:", vals)
print("y_train counts:", counts)

# 6) Verify row counts match
if X_train.shape[0] != y_train.shape[0]:
    print("⚠️  MISMATCH: X_train rows != y_train rows!")


# Prepare LightGBM parameter dictionary from wandb.config
params = {
    "objective":    config.objective,
    "boosting_type":config.boosting_type,
    "learning_rate":config.learning_rate,
    "num_leaves":   config.num_leaves,
    "max_depth":    config.max_depth,
    "subsample":    config.subsample,        # alias bagging_fraction
    "subsample_freq": config.subsample_freq, # alias bagging_freq
    "feature_fraction": config.feature_fraction,  # alias colsample_bytree:contentReference[oaicite:10]{index=10}
    "min_data_in_leaf": config.min_data_in_leaf,
    "min_sum_hessian_in_leaf": config.min_sum_hessian_in_leaf,
    "min_gain_to_split": config.min_gain_to_split,
    "lambda_l1":    config.lambda_l1,
    "lambda_l2":    config.lambda_l2,
    "max_bin":      config.max_bin,
    "metric":       config.metric,
    "seed":         config.seed,
    "verbosity":    config.verbosity  # control LightGBM's verbosity (set -1 to suppress info)
}
# params["scale_pos_weight"] = X_train.shape[0] - y_train.sum() / y_train.sum()
# params["metric"] = ["auc", "average_precision"]
# Handle class imbalance parameters to use only one of is_unbalance or scale_pos_weight:contentReference[oaicite:11]{index=11}
if config.is_unbalance:
    params["is_unbalance"] = True
    # If scale_pos_weight was set to something other than 1, ignore it in favor of is_unbalance
    params.pop("scale_pos_weight", None)
elif config.scale_pos_weight != 1.0:
    params["scale_pos_weight"] = config.scale_pos_weight
train_ds = lgb.Dataset(X_train, label=y_train, params=params)
val_ds   = lgb.Dataset(X_val, label=y_val, reference=train_ds, params=params)
# Train the LightGBM model with early stopping and evaluation logging
print("Training LightGBM model...")
# callbacks = [
#     lgb.early_stopping(stopping_rounds=config.early_stopping_rounds),
#     lgb.log_evaluation(config.eval_log_interval)  # log metrics every eval_log_interval rounds:contentReference[oaicite:12]{index=12}
# ]
callbacks = [
  lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, first_metric_only=True),
  lgb.log_evaluation(config.eval_log_interval),
]
bst = lgb.train(params, train_ds, num_boost_round=config.n_estimators,
                valid_sets=[train_ds, val_ds], valid_names=["train","val"],
                callbacks=callbacks)
# -----------------------------------
# Evaluate on train, val, test
# -----------------------------------
print("Evaluating model performance...")

# Convert to proper numpy arrays
y_train_arr = np.asarray(y_train)
y_val_arr   = np.asarray(y_val)
# If y_test was loaded as None or some odd object, this gives at least a 0-d array or empty array
y_test_arr  = np.asarray(y_test) if y_test is not None else np.array([])

# Predictions (you already have these)
train_preds_proba = bst.predict(X_train)
val_preds_proba   = bst.predict(X_val)
test_preds_proba  = bst.predict(X_test)

import numpy as np
from sklearn.metrics import f1_score, fbeta_score

# val_preds_proba: your array of shape (N,) with model.predict(X_val)
# y_val_arr: your ground-truth labels for validation, shape (N,)

def find_best_threshold(y_true, y_proba, beta=1.0, n_steps=100):
    """
    Sweep thresholds in [0,1] to find the one that maximizes the F-beta score.
    
    Returns:
      best_thresh: the threshold with highest F-beta
      best_score:  the corresponding best F-beta score
      all_scores:  list of (threshold, score) tuples
    """
    thresholds = np.linspace(0, 1, n_steps+1)
    best_score = -1.0
    best_thresh = 0.5
    all_scores = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        if beta == 1:
            score = f1_score(y_true, preds)
        else:
            score = fbeta_score(y_true, preds, beta=beta)
        all_scores.append((t, score))
        if score > best_score:
            best_score, best_thresh = score, t

    return best_thresh, best_score, all_scores

# — find best F1 threshold
best_f1_thresh, best_f1, _ = find_best_threshold(y_val_arr, val_preds_proba, beta=1, n_steps=100)

# — find best F2 threshold (if you want to weight recall more)
best_f2_thresh, best_f2, _ = find_best_threshold(y_val_arr, val_preds_proba, beta=2, n_steps=100)

print(f"Best F1={best_f1:.4f} at threshold {best_f1_thresh:.2f}")
print(f"Best F2={best_f2:.4f} at threshold {best_f2_thresh:.2f}")

# Log them to W&B
wandb.log({
    "best_val_f1": best_f1,
    "best_val_f1_threshold": best_f1_thresh,
    "best_val_f2": best_f2,
    "best_val_f2_threshold": best_f2_thresh
})


train_preds = (train_preds_proba  >= best_f1_thresh).astype(int)
val_preds   = (val_preds_proba    >= best_f1_thresh).astype(int)
test_preds  = (test_preds_proba   >= best_f1_thresh).astype(int)
class_names = ["Not Fraud", "Fraud"]

# Build confusion matrix panels correctly
conf_mat_train = wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_train_arr,
    preds=train_preds,
    class_names=class_names
)
conf_mat_val = wandb.plot.confusion_matrix(
    probs=None,
    y_true=y_val_arr,
    preds=val_preds,
    class_names=class_names
)

# Only if test labels exist
conf_mat_test = None
if y_test_arr.size > 0:
    conf_mat_test = wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test_arr,
        preds=test_preds,
        class_names=class_names
    )

# Log them
log_dict = {
    "conf_mat_train": conf_mat_train,
    "conf_mat_val": conf_mat_val
}
if conf_mat_test is not None:
    log_dict["conf_mat_test"] = conf_mat_test

wandb.log(log_dict)
# Compute metrics for train and val
train_acc, train_prec = accuracy_score(y_train_arr, train_preds), precision_score(y_train_arr, train_preds)
train_rec, train_f1   = recall_score(y_train_arr, train_preds), f1_score(y_train_arr, train_preds)
train_auc             = roc_auc_score(y_train_arr, train_preds_proba)

val_acc, val_prec     = accuracy_score(y_val_arr, val_preds), precision_score(y_val_arr, val_preds)
val_rec, val_f1       = recall_score(y_val_arr, val_preds), f1_score(y_val_arr, val_preds)
val_auc               = roc_auc_score(y_val_arr, val_preds_proba)

# Initialize test metrics as None
test_acc = test_prec = test_rec = test_f1 = test_auc = None
if y_test_arr.size > 0:
    try:
        test_acc  = accuracy_score(y_test_arr, test_preds)
        test_prec = precision_score(y_test_arr, test_preds)
        test_rec  = recall_score(y_test_arr, test_preds)
        test_f1   = f1_score(y_test_arr, test_preds)
        test_auc  = roc_auc_score(y_test_arr, test_preds_proba)
    except Exception as e:
        print(f"Warning: could not compute test metrics ({e}), skipping.")

# Log everything to W&B
wandb.log({
    "conf_mat_train": conf_mat_train,
    "conf_mat_val":   conf_mat_val,
    **({"conf_mat_test": conf_mat_test} if y_test_arr.size>0 else {}),
    # and re-log metrics so they align with the new threshold:
    "train_precision": precision_score(y_train_arr, train_preds),
    "train_recall":    recall_score(y_train_arr, train_preds),
    "train_f1":        f1_score(y_train_arr, train_preds),
    "val_precision":   precision_score(y_val_arr, val_preds),
    "val_recall":      recall_score(y_val_arr, val_preds),
    "val_f1":          f1_score(y_val_arr, val_preds),
    **({"test_precision":precision_score(y_test_arr, test_preds),
       "test_recall":   recall_score(y_test_arr, test_preds),
       "test_f1":       f1_score(y_test_arr, test_preds)} if y_test_arr.size>0 else {})
})

# Confusion matrices

# Classification report in summary (only if test labels exist)
if y_test_arr.size > 0:
    report = classification_report(y_test_arr, test_preds, target_names=class_names, digits=4)
    wandb.run.summary["classification_report"] = report
    print("Test Classification Report:\n", report)
else:
    print("No valid test labels found; skipping classification report.")


# -----------------------------------
# Save model and artifacts
# -----------------------------------
model_path = os.path.join(args.feature_dir, "lgbm_best_model.txt")
bst.save_model(model_path)
print(f"Model saved to {model_path}")
# Log the model as a W&B artifact for later retrieval
model_artifact = wandb.Artifact(name="lgbm_model", type="model")
model_artifact.add_file(model_path)
wandb.log_artifact(model_artifact)
# Optionally log feature files as artifact (if desired and if they exist on disk)
if not args.in_memory:
    data_artifact = wandb.Artifact(name="embeddings", type="dataset")
    for key, path in feature_paths.items():
        if os.path.isfile(path):
            data_artifact.add_file(path)
    wandb.log_artifact(data_artifact)

print("All done! Logged results to Weights & Biases.")
wandb.finish()
