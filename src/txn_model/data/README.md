# data/ Module

**Quickstart:**

1. **Preprocess your CSV**

   ```python
   from data.preprocessing import preprocess
   df, encoders = preprocess(
     "transactions.csv",
     cols_to_drop=[...],
     date_features=["unix_trans_time"],
     cat_features=[...],
     cont_features=[...]
   )
   ```

2. **Create the Dataset**

   ```python
   from data.dataset import TxnDataset
   ds = TxnDataset(
     df=df,
     group_key="cc_num",
     cat_features=[...],
     cont_features=[...],
     max_len=256  # optional
   )
   ```

3. **Build the DataLoader**

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

