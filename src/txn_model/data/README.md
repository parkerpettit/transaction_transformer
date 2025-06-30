````markdown
# data/ Module

Turn raw CSV transactions into Transformer-ready mini-batches in four simple steps.

## 1. Preprocessing (`preprocessing.py`)

- **Input:** Raw CSV with dates, categories, amounts, etc.
- **Actions:**
  1. Drop unused columns.
  2. Convert date/timestamp columns to numeric features.
  3. Fit & apply `LabelEncoder` to each categorical field.
- **Output:**
  - `df_processed` (sorted `pd.DataFrame` with only numeric columns)
  - `encoders` (`dict[str, LabelEncoder]`) for each categorical field

## 2. Dataset (`dataset.py`)

**Class:** `TxnDataset`

```python
TxnDataset(
    df: pd.DataFrame,            # sorted by group_key then time
    group_key: str,             # e.g. "cc_num"
    cat_fields: list[str],
    cont_fields: list[str],
    max_len: int | None = None  # history window (optional)
)
````

* **Initialization:**

  1. Reset DataFrame index for contiguous slicing.
  2. Convert `cat_fields` → `cat_array` (shape `[T, C]`).
     Convert `cont_fields` → `cont_array` (shape `[T, F]`).
  3. Record `group_starts` & `group_lengths` for each user/card.
  4. Build `cum_counts` so `__getitem__` can map sample index → (card, time-step).
* **`__getitem__(idx)`:**

  * Determine `group_id` & local step `t` via `cum_counts`.
  * Compute slice indices:

    ```python
    base = group_starts[group_id]
    seq_end = base + t + 1
    left = max(base, seq_end - max_len) if max_len else base
    ```
  * Slice:

    * `cat_context = cat_array[left:seq_end]` → `[seq_len, C]`
    * `cont_context = cont_array[left:seq_end]` → `[seq_len, F]`
    * `cat_target = cat_array[seq_end]` → `[C]`
    * `cont_target = cont_array[seq_end]` → `[F]`
  * Return:

    ```python
    inputs = {"cat": cat_context, "cont": cont_context}
    targets = {"tgt_cat": cat_target, "tgt_cont": cont_target}
    ```

## 3. Collation (`collate.py`)

**Function:** `collate_fn(batch)`

* **Input:** list of `(inputs, targets)` pairs from `__getitem__`.
* **Process:**

  1. Extract and pad `inputs["cat"]` & `inputs["cont"]` → `[B, max_seq, C]`, `[B, max_seq, F]`.
  2. Stack `targets["tgt_cat"]` → `[B, C]`; `targets["tgt_cont"]` → `[B, F]`.
  3. Build boolean `padding_mask` of shape `[B, max_seq]`.
* **Output:**

  ```python
  batch_inputs:  {"cat": Tensor[B, max_seq, C], "cont": Tensor[B, max_seq, F]}
  batch_targets: {"tgt_cat": Tensor[B, C], "tgt_cont": Tensor[B, F]}
  padding_mask:  BoolTensor[B, max_seq]
  ```

## 4. DataLoader Setup

```python
from torch.utils.data import DataLoader
from data.preprocessing import preprocess
from data.dataset import TxnDataset
from data.collate import collate_fn

# 1) Preprocess
df_processed, encoders = preprocess(
    "transactions.csv",
    cols_to_drop=[...],
    date_features=["unix_trans_time"],
    cat_fields=["merchant_id", "category_id", ...],
)

# 2) Dataset
ds = TxnDataset(
    df=df_processed,
    group_key="cc_num",
    cat_fields=[...],
    cont_fields=[...],
    max_len=256,  # optional window
)

# 3) DataLoader
loader = DataLoader(
    ds,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)
```

**Notes:**

* Use `encoders` at inference to map IDs back to labels.
* Update `cat_fields`/`cont_fields` lists when adding/removing features.

```
```
