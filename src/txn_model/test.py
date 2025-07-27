# import torch
# from data.dataset import TxnDataset, extract_embeddings
# from torch.utils.data import DataLoader
# from utils import load_ckpt

# print("Loading dfs")
# train_df, val_df, test_df, enc, cat_features, cont_features, scaler = torch.load("data/full_processed.pt",  weights_only=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Loading model")
# model, best_val, epoch = load_ckpt("data/big_legit_backbone.pt", device=device)
# print("Making train_ds")
# train_ds = TxnDataset(train_df, group_by="User", cat_features=cat_features, cont_features=cont_features, window=10, stride=10)
# val_ds = TxnDataset(val_df, group_by="User", cat_features=cat_features, cont_features=cont_features, window=10, stride=10)

# train_cache_path = "data/train_embedded_data"
# val_cache_path = "data/val_embedded_data"

# extract_embeddings(model, train_ds, train_cache_path, 128, device)
# extract_embeddings(model, val_ds, val_cache_path, 128, device)

# # x = torch.tensor([[0, 0], [0, 0]])
# # print(x)
# # y = torch.tensor([1, 1])
# # print(y)
# # x[0] = y
# # print(x)
# # y += 1
# # print(y)
# # print(x)
import pandas as pd

# 1) load your PR‐curve table
df = pd.read_csv("data/pr_curve.csv")
df = df[df["class"] == "fraud"]
# 2) compute F1 = 2·P·R / (P + R), guarding against zero‐division
df["f1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"]).replace(0, 1e-10)

# 3) find the row with the highest F1
best_idx = df["f1"].idxmax()
best_row = df.loc[best_idx]

print(f"Max F1 = {best_row['f1']:.4f} (precision={best_row['precision']:.4f}, recall={best_row['recall']:.4f})")
