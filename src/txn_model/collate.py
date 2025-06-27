import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(
  batch: list[
        tuple[
            dict[str, torch.Tensor],   # inputs_dict
            dict[str, torch.Tensor],   # target_cat_dict
            dict[str, torch.Tensor]    # target_cont_dict
        ]
    ]
) -> tuple[
    dict[str, torch.Tensor],       # batch_inputs
    dict[str, torch.Tensor],       # batch_targets_cat
    dict[str, torch.Tensor],       # batch_targets_cont
    torch.BoolTensor               # padding_mask
]:
    """
    Args:
        batch (list of tuples):  
            Each element in the list is one training example, represented as a 3-tuple:
            
            - inputs_dict (dict[str, Tensor]):  
              Maps each input field name (e.g. "cat_merchant_id", "cont_amt", …) to a 1D Tensor
              of variable length (the transaction history for that feature).

            - target_cat_dict (dict[str, Tensor]):  
              Maps each categorical target field name (e.g. "tgt_cat_merchant_id") to a scalar
              LongTensor holding the next-transaction category ID.

            - target_cont_dict (dict[str, Tensor]):  
              Maps each continuous target field name (e.g. "tgt_cont_amt") to a scalar
              FloatTensor holding the next-transaction value.

    Returns:
        batch_inputs (dict[str, Tensor]):  
            Same keys as inputs_dict, but each Tensor is now of shape
            (batch_size, max_seq_len), padded to the longest sequence in the batch.

        batch_tgts_cat (dict[str, Tensor]):  
            Same keys as target_cat_dict, but each Tensor is of shape (batch_size,).

        batch_tgts_cont (dict[str, Tensor]):  
            Same keys as target_cont_dict, but each Tensor is of shape (batch_size,).

        padding_mask (BoolTensor):  
            A mask of shape (batch_size, max_seq_len) where True indicates real data
            and False indicates padding positions.
    """

    input_keys = list(batch[0][0].keys())
    tgt_cat_keys = list(batch[0][1].keys())
    tgt_cont_keys = list(batch[0][2].keys())
    rep_key = input_keys[0]   # for length/truncation

    rep_seqs = [sample[0][rep_key] for sample in batch]
    lengths = [len(s) for s in rep_seqs]
    max_seq_len = max(lengths)
    batch_size = len(batch)


    # Pad all training tensors to maximum length with 0s
    batch_inputs: dict[str, torch.tensor] = {
        key: pad_sequence(
           [sample[0][key] for sample in batch],
           batch_first=True,
           padding_value=0
        )
        for key in input_keys                                
    }
  
    # Collect categorical labels into a tensor
    batch_tgts_cat: dict[str, torch.tensor] = {
      key: torch.stack([sample[1][key] for sample in batch], dim=0)
      for key in tgt_cat_keys
    }

    # Collect continuous labels into a tensor
    batch_tgts_cont: dict[str, torch.tensor] = {
      key: torch.stack([sample[2][key] for sample in batch], dim=0)
      for key in tgt_cont_keys
    }

    # Create padding mask
    device = next(iter(batch_inputs.values())).device
    lengths_tensor = torch.tensor(lengths, device=device)
    idx = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    padding_mask = (idx < lengths_tensor.unsqueeze(1)) # true where there's real tokens, false where padded

    return batch_inputs, batch_tgts_cat, batch_tgts_cont, padding_mask






