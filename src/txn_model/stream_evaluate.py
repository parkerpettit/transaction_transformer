import csv
import numpy as np
from itertools import islice
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def stream_metrics(feat_csv, preds_txt, chunk_size=100_000):
    # read labels in chunks
    y_true_chunks = []
    with open(feat_csv, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for lines in iter(lambda: list(islice(reader, chunk_size)), []):
            y_true_chunks.append([int(row[-1]) for row in lines])
    y_true = np.concatenate(y_true_chunks)

    # read all probabilities
    y_proba = np.fromfile(preds_txt, dtype=np.float64, sep="\n")
    y_pred  = (y_proba >= 0.5).astype(int)

    print("AUC:      ", roc_auc_score(y_true, y_proba))
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:   ", recall_score(y_true, y_pred))
    print("F1:       ", f1_score(y_true, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification report:\n",
          classification_report(y_true, y_pred, target_names=["non-fraud","fraud"]))

stream_metrics("features/test_features.csv", "test_probs.txt")
