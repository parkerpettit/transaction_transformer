import argparse
import numpy as np
import lightgbm as lgb
import wandb
from wandb.integration.lightgbm import wandb_callback, log_summary
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.metrics import ConfusionMatrixDisplay 

# ###################################################################################
# #####    UNCOMMENT THIS BLOCK TO COLLECT AND SAVE EMBEDDINGS AS NPY FILES #########
# ###################################################################################
# import torch
# from torch.utils.data import DataLoader
# from data.dataset import TxnDataset, collate_fn
# from pathlib import Path
# from utils import load_ckpt
# model_path = Path("data", "pretrained_backbone.pt") # model thats only seen legit behavior, path will change on next run
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# backbone, _, _ = load_ckpt(model_path)
# model = backbone.to(device).eval()

# data_path = Path("data", "all_data_on_legit_encoders.pt")   # includes fraudulent transactions
# train_df, val_df, test_df, enc, cat_feats, cont_feats, scaler = torch.load(data_path, weights_only=False)
# window_size, stride = 10, 5

# train_dataset = TxnDataset(train_df, cat_feats[0], cat_feats, cont_feats, window_size, stride)
# val_dataset = TxnDataset(val_df, cat_feats[0], cat_feats, cont_feats, window_size, stride)
# test_dataset = TxnDataset(test_df, cat_feats[0], cat_feats, cont_feats, window_size, stride)

# batch_size = 256
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# def save_embeddings(model, loader, device, suffix: str):
#     all_embs = []
#     all_labels = []
#     with torch.no_grad():
#         for batch in loader:
#             inputs = {
#                 "cat":       batch["cat"].to(device),
#                 "cont":      batch["cont"].to(device),
#                 "pad_mask":  batch["pad_mask"].to(device),
#             }
#             labels = batch["label"].to(device)
#             model.to(device)
#             # now safe to call
#             out = model(**inputs, mode="lightgbm")
#                         # out = model(**inputs, mode="lightgbm")
            
#             emb = out.cpu()
#             all_embs.append(emb)
#             all_labels.append(labels.cpu())
        
#         # Concatenate all batches into final arrays
#         X = torch.cat(all_embs, dim=0).numpy()         # shape = [N, D]
#         Y = torch.cat(all_labels, dim=0).numpy().astype(int)  # shape = [N]
#     X_filename = f"X_{suffix}.npy"
#     Y_filename = f"Y_{suffix}.npy"
#     np.save(Path("data", X_filename), X)
#     np.save(Path("data", Y_filename), Y)
#     print(f"Files saved as {X_filename} and {Y_filename}.")


# save_embeddings(model, train_loader, device, "train")
# save_embeddings(model, val_loader, device, "val")
# save_embeddings(model, test_loader, device, "test")
# ##################################################################################
# ################################ END BLOCK #######################################
# ##################################################################################
# 1. Configuration and Setup
parser = argparse.ArgumentParser(description="Train LightGBM on Transformer embeddings for fraud detection")
parser.add_argument("--wandb_project",        type=str, default="lightgbm", help="Weights & Biases project name")
parser.add_argument("--wandb_run_name",       type=str, default="lgbm_run", help="Weights & Biases run name")
parser.add_argument("--learning_rate",        type=float, default=0.05, help="Learning rate")
parser.add_argument("--num_leaves",           type=int,   default=127,  help="Max number of leaves per tree")
parser.add_argument("--max_depth",            type=int,   default=-1,   help="Max tree depth (-1 for no limit)")
parser.add_argument("--n_estimators",         type=int,   default=1000, help="Number of boosting rounds")
parser.add_argument("--subsample",            type=float, default=1.0,  help="Subsample fraction")
parser.add_argument("--subsample_freq",       type=int,   default=0,    help="Subsample frequency")
parser.add_argument("--feature_fraction",     type=float, default=1.0,  help="Feature fraction (colsample_bytree)")
parser.add_argument("--min_data_in_leaf",     type=int,   default=20,   help="Min data in leaf")
parser.add_argument("--min_sum_hessian_in_leaf", type=float, default=1e-3, help="Min sum Hessian in leaf")
parser.add_argument("--min_gain_to_split",    type=float, default=0.0,  help="Min gain to split")
parser.add_argument("--lambda_l1",            type=float, default=0.0,  help="L1 regularization (reg_alpha)")
parser.add_argument("--lambda_l2",            type=float, default=0.0,  help="L2 regularization (reg_lambda)")
parser.add_argument("--boosting_type",        type=str,   default="gbdt",choices=["gbdt","dart","goss"], help="Boosting type")
parser.add_argument("--objective",            type=str,   default="binary", help="Objective function")
parser.add_argument("--early_stopping_rounds",type=int,   default=60,   help="Early stopping rounds")
args = parser.parse_args()


from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
def evaluate_recall_at_fprs(y_true, y_pred_proba, fpr_thresholds, granularity = 2):
  results = []
  for fpr_threshold in fpr_thresholds:
    best_recall = 0
    best_threshold = 0
    #for binary search
    bot = 0
    top = 1
    while abs(top-bot) >= 10*.1**granularity:
      threshold = round(((top+bot)/2), granularity)
      y_pred = (y_pred_proba >= threshold).astype(int)
      tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0,1]).ravel()
      recall = tp/(tp+fn) if (tp+fn) != 0 else 0
      fpr = fp/(fp+tn) if (fp+tn) != 0 else 0
      if fpr < fpr_threshold and recall > best_recall:
        best_recall = recall
        best_threshold = threshold
      
      if fpr >= fpr_threshold:
        bot = threshold
      else:
        top = threshold
    
    results.append({
        'threshold': best_threshold,
        'best_recall': best_recall,
        'fpr_limit': fpr_threshold
    })
  return pd.DataFrame(results)


# Initialize W&B
wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    config=vars(args)
)
config = wandb.config
# 2. Load embeddings (pre-saved .npy files)
X_train = np.load(Path("data", "X_train.npy"))
Y_train = np.load(Path("data", "Y_train.npy"))
X_val   = np.load(Path("data", "X_val.npy"))
Y_val   = np.load(Path("data", "Y_val.npy"))
X_test  = np.load(Path("data", "X_test.npy"))
Y_test  = np.load(Path("data", "Y_test.npy"))

# Print shapes of embedding matrices
print("Shapes:")
for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
    print(f"  {name}: {X.shape}")

def summarize_labels(name, Y):
    unique, counts = np.unique(Y, return_counts=True)
    total = counts.sum()
    print(f"\n{name} labels:")
    for lbl, cnt in zip(unique, counts):
        pct = cnt / total * 100
        print(f"  Label {lbl}: {cnt} samples ({pct:.2f}%)")
    print(f"  Total: {total} samples")

# Print fraud/non-fraud stats
summarize_labels("Y_train", Y_train)
summarize_labels("Y_val", Y_val)
summarize_labels("Y_test", Y_test)


# 3. Create & train the LGBMClassifier with W&B callback
clf = lgb.LGBMClassifier(
    objective          = config.objective,
    learning_rate      = config.learning_rate,
    num_leaves         = config.num_leaves,
    max_depth          = config.max_depth,
    n_estimators       = config.n_estimators,
    subsample          = config.subsample,
    subsample_freq     = config.subsample_freq,
    colsample_bytree   = config.feature_fraction,
    min_child_samples  = config.min_data_in_leaf,
    min_child_weight   = config.min_sum_hessian_in_leaf,
    min_split_gain     = config.min_gain_to_split,
    reg_alpha          = config.lambda_l1,
    reg_lambda         = config.lambda_l2,
    boosting_type      = config.boosting_type,
    is_unbalance       = True,
    random_state       = 42,
    verbosity          = 2
)

clf.fit(
    X_train, Y_train,
    eval_set       = [(X_train, Y_train), (X_val, Y_val)],
    eval_names     = ["train","val"],
    eval_metric    = ["average_precision", "auc"],
    callbacks      = [wandb_callback(), lgb.early_stopping(config.early_stopping_rounds)],
)

# 4. Log model summary & save checkpoint
log_summary(clf.booster_, save_model_checkpoint=True)

# 5. Evaluate on test set
y_prob = clf.predict_proba(X_test)[:, 1]


# 6. Log metrics to W&B
# metrics = {
#     "test/accuracy":           accuracy_score(Y_test, y_pred),
#     "test/precision":          precision_score(Y_test, y_pred),
#     "test/recall":             recall_score(Y_test, y_pred),
#     "test/f1":                 f1_score(Y_test, y_pred),
#     "test/roc_auc":            roc_auc_score(Y_test, y_prob),
#     "test/average_precision":  average_precision_score(Y_test, y_prob),
# }
# wandb.log(metrics)

# 7. Plot & log PDF of probabilities
counts, bin_edges = np.histogram(y_prob, bins=100, density=True)
centers = (bin_edges[:-1] + bin_edges[1:]) / 2
fig1, ax1 = plt.subplots()
ax1.plot(centers, counts, drawstyle='steps-mid')
ax1.set(title="PDF of Predicted Probabilities", xlabel="Probability", ylabel="Density")
wandb.log({"pred_prob_pdf": wandb.Image(fig1)})

# 8. Plot & log confusion matrix
fpr_thresholds = [.9,.8,.7,.6,.5,.4,.3,.2,.1]
results = evaluate_recall_at_fprs(Y_test, y_prob, fpr_thresholds)
for threshold in results["threshold"]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(Y_test, y_pred)
    fig2, ax2 = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-fraud","fraud"])
    disp.plot(ax=ax2, cmap="Blues", values_format="d")
    ax2.set(title="Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(fig2)})

# 9. Log classification report
report = classification_report(Y_test, y_pred, target_names=["non-fraud","fraud"], output_dict=True)
wandb.log({"classification_report": report})

# 10. Finish W&B run
wandb.finish()


from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
def evaluate_recall_at_fprs(y_true, y_pred_proba, fpr_thresholds, granularity = 10):
  results = []
  for fpr_threshold in fpr_thresholds:
    best_recall = 0
    best_threshold = 0
    #for binary search
    bot = 0
    top = 1
    while abs(top-bot) >= 10*.1**granularity:
      threshold = round(((top+bot)/2), granularity)
      y_pred = (y_pred_proba >= threshold).astype(int)
      tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0,1]).ravel()
      recall = tp/(tp+fn) if (tp+fn) != 0 else 0
      fpr = fp/(fp+tn) if (fp+tn) != 0 else 0
      if fpr < fpr_threshold and recall > best_recall:
        best_recall = recall
        best_threshold = threshold
      
      if fpr >= fpr_threshold:
        bot = threshold
      else:
        top = threshold
    
    results.append({
        'threshold': best_threshold,
        'best_recall': best_recall,
        'fpr_limit': fpr_threshold
    })
  return pd.DataFrame(results)


