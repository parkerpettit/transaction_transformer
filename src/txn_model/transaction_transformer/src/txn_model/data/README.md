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
   from data.collate import collate_fn

   loader = DataLoader(
     ds,
     batch_size=64,
     shuffle=True,
     collate_fn=collate_fn
   )
   ```

4. **Train**  Iterate over `loader` to get `(inputs, targets, mask)` batches.

---

* Use `encoders` at inference to map IDs back to original labels.
* Adjust `max_len`, `batch_size`, and features lists as needed.
