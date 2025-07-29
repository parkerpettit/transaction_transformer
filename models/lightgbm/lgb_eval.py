#!/usr/bin/env python
import argparse
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, auc, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import itertools
import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


## EVALUATION HELPERS FOR LIGHTGBM ##
def evaluate_model_helper(X_test, y_test, lr, exp_info="", viz=False):
    fpr_thresholds = [
        val / 100.0
        for val in [0.4, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    ]
    y_pred_proba = lr.predict_proba(X_test)[:, 1] * 1000
    plt.figure(figsize=(15, 15))
    plt.suptitle(exp_info)

    j = 1
    score_thresholds = []
    recall_values = []
    final_cnf_matrix = None

    for fpr_threshold in fpr_thresholds:
        min_score_threshold = None
        max_recall = 0.0
        max_recall_score_threshold = None
        for i in range(1001):
            y_test_predictions_high_recall = y_pred_proba >= i
            cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)

            TP = cnf_matrix[1, 1]
            FN = cnf_matrix[1, 0]
            FP = cnf_matrix[0, 1]
            TN = cnf_matrix[0, 0]

            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            fpr = FP / (FP + TN) if (FP + TN) != 0 else 0

            if fpr <= fpr_threshold:
                min_score_threshold = i
                max_recall = recall
                max_recall_score_threshold = i
                final_cnf_matrix = cnf_matrix
                break

        score_thresholds.append(max_recall_score_threshold)
        
        if max_recall==None:
            print("here")
        
        recall_values.append(max_recall)

        if viz == True:
            plt.subplot(4, 3, j)
            j += 1

            plot_confusion_matrix(
                final_cnf_matrix,
                classes=["No Fraud", "Fraud"],
                title=f"Threshold = {max_recall_score_threshold}\nFPR<={(fpr_threshold * 100):.2f}% \nRecall={max_recall:.4f}",
            )
    results_fpr = {
        "score_threshold": score_thresholds,
        "fpr %": [f"{fpr * 100.0:.2f}" for fpr in fpr_thresholds],
        "recall %": [f"{(recall * 100.0):.2f}" for recall in recall_values],
    }
    results_fpr_table = pd.DataFrame(results_fpr).sort_values(by="fpr %")
    return results_fpr_table


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def calculate_partial_auc(y_test, probs, fpr_threshold=0.001):
    """
    Calculates the AUC for a specific range of FPR (e.g., FPR <= 0.001).

    Parameters:
        y_test (array-like): True binary labels.
        probs (array-like): Predicted probabilities for the positive class.
        fpr_threshold (float): Maximum value of FPR for which to calculate the AUC.

    Returns:
        float: Partial AUC for FPR <= fpr_threshold.
    """
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)

    # Filter the range for FPR <= fpr_threshold
    mask = fpr <= fpr_threshold
    filtered_fpr = fpr[mask]
    filtered_tpr = tpr[mask]

    # Compute the partial AUC using trapezoidal rule
    partial_auc = np.trapz(filtered_tpr, filtered_fpr)
    return partial_auc


def get_best_params(X_train, y_train):
    def objective(trial):
        # Define the parameter search space
        param = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.7, 0.9),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.7, 0.9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        }

        # Prepare the dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)

        # Perform cross-validation with early stopping
        cv_results = lgb.cv(
            params=param,
            train_set=train_data,
            nfold=3,
            metrics=["auc"],
            # early_stopping_rounds=50,
            seed=42,
        )

        # Get the best AUC score from cross-validation (mean of validation AUC scores)
        return max(cv_results["valid auc-mean"])

    # Run the optimization process with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Retrieve the best parameters
    best_params = study.best_params

    return best_params


def save_params(best_params, filepath):
    with open(filepath, "w") as file:
        json.dump(best_params, file)


## UTILS ##
def get_fraud_percentage(counter):
    num_fraud = counter[1]
    num_nonfraud = counter[0]
    total = num_fraud + num_nonfraud
    print(
        f"{(num_fraud * 100.0)/total:.4f}% fraud -- {num_fraud} fraud, {num_nonfraud} nonfraud, {total} total"
    )


def duplicate_minority_class(dataset, N):
    fraud_class = dataset[dataset["Class"] == 1]
    nonfraud_class = dataset[dataset["Class"] == 0]

    duplicated_fraud_class = pd.concat([fraud_class] * N, ignore_index=True)
    # Concatenate the duplicated fraud class back to the original dataset
    df_duplicated = pd.concat(
        [nonfraud_class, duplicated_fraud_class], ignore_index=True
    )
    return df_duplicated


# Function to apply SMOTE only to the fraud class
def apply_smote_to_fraud(X, y, target_fraud_ratio):
    # Calculate current number of fraud samples
    fraud_count = y.sum()
    non_fraud_count = len(y) - fraud_count

    # Desired number of fraud samples
    target_fraud_count = int(
        non_fraud_count * target_fraud_ratio / (1 - target_fraud_ratio)
    )

    # Sampling strategy for SMOTE
    smote_strategy = {1: target_fraud_count, 0: non_fraud_count}

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


## OTHER -- MAY GET RID LATER ##
def plot_two_features(all_train, feature_x, feature_y, title_text="", color_col=None):
    """
    Create a scatter plot comparing two features colored by fraud status.

    Args:
        all_train (DataFrame): Dataset containing the features and the "isFraud" column.
        feature_x (str): Name of the first feature.
        feature_y (str): Name of the second feature.
        title_text (str): Title for the plot.

    Returns:
        plotly.graph_objects.Figure: A scatter plot figure.
    """
    fig = px.scatter(
        all_train,
        x=feature_x,
        y=feature_y,
        color=color_col,
        opacity=0.5,
        title=title_text,
    )
    fig.update_layout(
        title_font_color="white",
        legend_title_font_color="yellow",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font_color="white",
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color="Grey")),
        selector=dict(mode="markers"),
    )
    return fig
# -----------------------------------------------------------
# ASSUMPTIONS:
#   - You already defined in the same module (or imported):
#       evaluate_model_helper, plot_confusion_matrix,
#       calculate_partial_auc, get_best_params
#   - Target column name provided via --target_col (default: is_fraud)
# -----------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True,
                    help="Path to CSV or torch-saved .pt with (train_df, val_df, ...).")
    ap.add_argument("--target_col", type=str, default="is_fraud")
    ap.add_argument("--wandb_project", type=str, default="lgbm_baseline")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optuna_trials", type=int, default=0,
                    help="If >0 run Optuna to tune and then train final model.")
    ap.add_argument("--model_out", type=str, default="lgbm_model.txt")
    return ap.parse_args()


def load_data(path, target_col):
    if path.endswith(".pt"):
        # Expect torch.save((train_df, val_df, ...), path)
        import torch
        obj = torch.load(path, weights_only=False)
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            train_df, val_df = obj[0], obj[1]
            df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
        else:
            raise ValueError(".pt file does not contain (train_df, val_df, ...)")
    else:
        df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    return df


def train_lgbm(X_train, y_train, X_val, y_val, params):
    clf = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        **params
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "average_precision"],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    return clf


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    wandb.init(project=args.wandb_project,
               name=args.wandb_run_name,
               config=vars(args),
               reinit=False)

    df = load_data(args.data_path, args.target_col)

    y = df[args.target_col].astype(int).values
    X = df.drop(columns=[args.target_col])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # Base params (you can expand)
    base_params = dict(
        learning_rate=0.05,
        num_leaves=63,
        n_estimators=1200,
        feature_fraction=0.9,
        subsample=0.9,
        subsample_freq=1,
        min_data_in_leaf=40,
        lambda_l1=0.0,
        lambda_l2=0.0,
        max_depth=-1,
        verbose=-1,
    )

    if args.optuna_trials > 0:
        # Use existing helper
        tuned = get_best_params(X_train, y_train)
        base_params.update(tuned)
        wandb.log({"tuning/best_params": json.dumps(tuned)}, commit=False)

    clf = train_lgbm(X_train, y_train, X_val, y_val, base_params)

    # Probabilities
    probs = clf.predict_proba(X_val)[:, 1]
    preds_05 = (probs >= 0.5).astype(int)

    # Global metrics
    roc_auc = roc_auc_score(y_val, probs)
    pr_auc = average_precision_score(y_val, probs)

    # Partial AUC low-FPR slices (examples)
    pAUC_001 = calculate_partial_auc(y_val, probs, fpr_threshold=0.001)
    pAUC_0005 = calculate_partial_auc(y_val, probs, fpr_threshold=0.0005)

    # PR curve points for optional logging
    prec_arr, rec_arr, thr_pr = precision_recall_curve(y_val, probs)
    pr_points = list(zip(map(float, rec_arr), map(float, prec_arr)))

    # Baseline confusion matrix (thr=0.5)
    cm_05 = confusion_matrix(y_val, preds_05, labels=[0, 1])

    # Use your helper to get FPR-constrained recall table
    # NOTE: helper multiplies probs by 1000 internally and sweeps integer thresholds.
    fpr_table = evaluate_model_helper(X_val, y_val, clf, exp_info="Baseline LGBM", viz=False)

    # Derive best F1 threshold (scan unique probs)
    # (quick inline implement—does not conflict with your functions)
    order = np.argsort(probs)
    unique_thr = np.unique(probs)
    best_f1 = -1
    best_f1_thr = 0.5
    for t in unique_thr:
        p = (probs >= t).astype(int)
        tp = ((p == 1) & (y_val == 1)).sum()
        fp = ((p == 1) & (y_val == 0)).sum()
        fn = ((p == 0) & (y_val == 1)).sum()
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = t

    preds_f1 = (probs >= best_f1_thr).astype(int)
    cm_f1 = confusion_matrix(y_val, preds_f1, labels=[0, 1])
    tp = ((preds_f1 == 1) & (y_val == 1)).sum()
    fp = ((preds_f1 == 1) & (y_val == 0)).sum()
    fn = ((preds_f1 == 0) & (y_val == 1)).sum()
    tn = ((preds_f1 == 0) & (y_val == 0)).sum()
    precision_f1 = tp / (tp + fp) if (tp + fp) else 0.0
    recall_f1 = tp / (tp + fn) if (tp + fn) else 0.0

    # Log scalar metrics
    wandb.log({
        "val/roc_auc": roc_auc,
        "val/pr_auc": pr_auc,
        "val/partial_auc_fpr<=0.001": pAUC_001,
        "val/partial_auc_fpr<=0.0005": pAUC_0005,
        "val/best_f1": best_f1,
        "val/best_f1_threshold": best_f1_thr,
        "val/best_f1_precision": precision_f1,
        "val/best_f1_recall": recall_f1,
        "val/cm_default0.5_tp": int(cm_05[1,1]),
        "val/cm_default0.5_fp": int(cm_05[0,1]),
        "val/cm_default0.5_tn": int(cm_05[0,0]),
        "val/cm_default0.5_fn": int(cm_05[1,0]),
    }, commit=False)

    # Log FPR table
    wandb.log({"val/fpr_recall_table": wandb.Table(dataframe=fpr_table)}, commit=False)

    # Plot & log confusion matrices (0.5 and best F1)
    def plot_and_log_cm(cm, title, key):
        from itertools import product
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["No Fraud","Fraud"])
        ax.set_yticklabels(["No Fraud","Fraud"])
        thresh = cm.max()/2
        for i,j in product(range(2), range(2)):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
        fig.tight_layout()
        wandb.log({key: wandb.Image(fig)}, commit=False)
        plt.close(fig)

    f1_text = f"F1 thr={best_f1_thr:.6f}\nF1={best_f1:.6f} P={precision_f1:.6f} R={recall_f1:.6f}"
    plot_and_log_cm(cm_05, "Confusion Matrix (thr=0.5)", "val/cm_thr_0_5")
    plot_and_log_cm(cm_f1, f1_text, "val/cm_best_f1")

    # ROC & PR curves (static) + points
    fpr, tpr, _ = roc_curve(y_val, probs)
    prec_curve, rec_curve, _ = precision_recall_curve(y_val, probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.6f}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend()
    wandb.log({"val/roc_curve_static": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(rec_curve, prec_curve, label=f"PR AUC={pr_auc:.6f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend()
    wandb.log({"val/pr_curve_static": wandb.Image(fig)}, commit=False)
    plt.close(fig)

    # Save model
    clf.booster_.save_model(args.model_out)
    art = wandb.Artifact("lgbm_model", type="model")
    art.add_file(args.model_out)
    wandb.log_artifact(art)

    wandb.log({"_end": 1}, commit=True)
    wandb.finish()


if __name__ == "__main__":
    main()
