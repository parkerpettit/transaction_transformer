# `data/` Module

This folder contains everything you need to turn raw CSV transaction data into mini‑batches ready for training your Transformer.  The high‑level pipeline is:

1. **Preprocessing** (`preprocessing.py`)

   - Read your CSV into a Pandas DataFrame.
   - Drop irrelevant columns, convert date fields to numerical features, and apply `LabelEncoder` to your categorical fields.
   - Returns:
     - `df_processed` (`pd.DataFrame`): all features numeric and ready for sequence building.
     - `encoders` (`dict[str, LabelEncoder]`): mapping from feature names to fitted encoders (for reverse lookup or inference).

2. **Build Examples** (`build_examples.py`)

   - Call `build_train_examples(df_processed, group_key, cat_fields, cont_fields)` to turn each user’s transaction history into a list of training examples.
   - Each example is a `dict` containing:
     - `cat_<field>` / `cont_<field>` → lists of past values (the context sequence).
     - `tgt_cat_<field>` / `tgt_cont_<field>` → the next-transaction targets (scalars).

3. **Dataset** (`dataset.py`)

   - Wrap the list of example dicts in `TxnDataset`, a standard PyTorch `Dataset` with `__len__` and `__getitem__`.

4. **Batching**

   - **Sampler** (`samplers.py`)
     - Use `AutoBucketSampler` (or a custom `BucketBatchSampler`) to group sequences of similar lengths into the same proto‑buckets, minimizing padding.
   - **Collate Function** (`collate.py`)
     - Pass `collate_fn` to your `DataLoader` to:
       1. Pad all variable‑length sequences in the batch to the maximum length.
       2. Build a boolean padding mask.
       3. Stack scalar targets into `(batch_size,)` tensors.

5. **DataLoader**

   ```python
   from torch.utils.data import DataLoader
   from txn_model.data.dataset import TxnDataset
   from txn_model.data.collate import collate_fn
   from txn_model.data.samplers import AutoBucketSampler

   # 1) Preprocess
   df_processed, encoders = preprocessing('transactions.csv', cols_to_drop=..., date_features=..., cat_fields=...)

   # 2) Build examples
   examples = build_train_examples(df_processed, group_key='cc_num', cat_fields=..., cont_fields=...)

   # 3) Dataset
   ds = TxnDataset(examples)

   # 4) Sampler
   lengths = [ len(ex['cat_merchant_id']) for ex in examples ]
   sampler = AutoBucketSampler(lengths, batch_size=32, bucket_size_multiplier=50)

   # 5) DataLoader
   loader = DataLoader(
       ds,
       batch_sampler=sampler,
       collate_fn=collate_fn
   )
   ```

---

**Tips:**

- You don’t need to write separate column‑mapping code—`preprocessing.py` returns the encoders you’ll need to translate numeric IDs back to labels at inference time.
- Adjust the `bucket_size_multiplier` to balance padding efficiency vs. within‑batch randomness.
- If you update field names or add new features, just update the argument lists in the preprocessing and example‑building calls.

