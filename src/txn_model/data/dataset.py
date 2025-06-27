import torch
from torch.utils.data import Dataset

class txnDataset(Dataset):
    """
      A Dataset for sequence-to-next-row training.

      Takes as input a list of dictionaries. Each dictionary in the list of examples represents a
      sequence of transactions made by one user. Naming conventions (handled by the preprocessing
      pipeline) are expected in the format:

          "cat_{field}" for field in categorical fields
          "cont_{field}" for field in continuous fields
      where field is the name of the field in the original dataset.

      Target (tgt) fields are the fields of the next row in the sequence, which is handled by the preprocessing
      pipeline. tgt columns must appear in each example under the prefixed keys:

          "tgt_cat_{field} for every field in cat_fields"
          "tgt_cont_{field} for every field in cont_fields"

      One example dict might look like:

          {
              "cat_merchant" : ("34", "42", "19"), # Nike, tgt, Costco, mapped to ID numbers
              "cat_category" : ("25", "81", "15"), # shoes, shoes, groceries, mapped to ID numbers
              "cont_amount": (100, 120, 200), # amount spent at Nike, tgt, Costco
              "tgt_cat_merchant: "11", # Adidas, mapped to ID number
              "tgt_cont_category": "32", # apparel, mapped to ID number
              "cont_tgt_amount": 200, # tgt spends $200 at Adidas
          }
      This example represents a sequence where a single user spends $100 at Nike, then
      $120 on tgt, and $200 at Costco. Our end goal is to predict the next transaction at Adidas.

    """
    def __init__(self, examples: list[dict]):
      """
      Initializes the txnDataset. Requires data to be preprocessed into the format specified in
      the txnDataset class docstring.
      """
      self.examples = examples
      sample = examples[0]
      self.cat_fields = [key for key in sample.keys() if key.startswith("cat_")]
      self.cont_fields = [key for key in sample.keys() if key.startswith("cont_")]
      self.tgt_cat_fields = [key for key in sample.keys() if key.startswith("tgt_cat_")]
      self.tgt_cont_fields = [key for key in sample.keys() if key.startswith("tgt_cont_")]
      if not self.cat_fields:
          raise ValueError("Missing categorical fields")
      if not self.cont_fields:
          raise ValueError("Missing continuous fields")
      if not self.tgt_cat_fields:
          raise ValueError("Missing tgt cat fields")
      if not self.tgt_cont_fields:
          raise ValueError("Missing tgt cont fields")

    def __len__(self):
      """
      Returns the number of examples in the dataset.
      """
      return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[
        dict[str, torch.Tensor],  # inputs  (sequences)
        dict[str, torch.Tensor],  # cat targets (scalars)
        dict[str, torch.Tensor],  # cont targets (scalars)
    ]:
      """
      Returns one training sample as tensors.
      """
      raw = self.examples[idx]

      inputs: dict[str, torch.Tensor] = {}
      for k in self.cat_fields:
        inputs[k] = torch.tensor(raw[k], dtype=torch.long)
      for k in self.cont_fields:
        inputs[k] = torch.tensor(raw[k], dtype=torch.float32)

      tgt_cat = {k: torch.tensor(raw[k], dtype=torch.long) for k in self.tgt_cat_fields}
      tgt_cont = {k: torch.tensor(raw[k], dtype=torch.float32) for k in self.tgt_cont_fields}

      return inputs, tgt_cat, tgt_cont


