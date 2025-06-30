import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

def collate_fn(
    batch: list[
        tuple[
            dict[str, Tensor],  # inputs: {"cat": [seq,C], "cont": [seq,F]}
            dict[str, Tensor],  # targets: {"tgt_cat": [C], "tgt_cont": [F]}
        ]
    ]
) -> tuple[
    dict[str, Tensor],  # batch_inputs: {"cat": [B, max_seq, C], "cont": [B, max_seq, F]}
    dict[str, Tensor],  # batch_targets: {"tgt_cat": [B, C], "tgt_cont": [B, F]}
    Tensor              # padding_mask: [B, max_seq]
]:
    """
    Collate a list of transaction-sequence examples into a padded batch.

    Each element of 'batch' is a tuple (inputs, targets), where:
      - inputs is a dict with
          "cat":   LongTensor [seq_len, C]  (categorical feature sequences)
          "cont":  FloatTensor [seq_len, F]  (continuous feature sequences)
      - targets is a dict with
          "tgt_cat":  LongTensor [C]         (categorical next-txn features)
          "tgt_cont": FloatTensor [F]        (continuous next-txn features)

    This function will:
      1. Stack all target tensors into shape [B, C] and [B, F].
      2. Pad each variable-length sequence in 'cat' and 'cont' up to
         the batch's maximum sequence length (max_seq), producing:
           cat_padded  -> LongTensor  [B, max_seq, C]
           cont_padded -> FloatTensor [B, max_seq, F]
      3. Build a boolean 'padding_mask' of shape [B, max_seq], where
         True indicates a real token and False indicates padding.
      4. Return three values:
           - batch_inputs:  {"cat": cat_padded, "cont": cont_padded}
           - batch_targets: {"tgt_cat": tgt_cat_batch, "tgt_cont": tgt_cont_batch}
           - padding_mask:  BoolTensor [B, max_seq]
    """
    # 1) unpack
    cat_seqs  = [sample[0]["cat"]  for sample in batch]   # list of [seq_i, C]
    cont_seqs = [sample[0]["cont"] for sample in batch]   # list of [seq_i, F]
    tgt_cat   = torch.stack([sample[1]["tgt_cat"] for sample in batch], dim=0)  # [B, C]
    tgt_cont  = torch.stack([sample[1]["tgt_cont"] for sample in batch], dim=0) # [B, F]

    # 2) pad the variable-length sequences to the same max_seq length
    #    pad_sequence will output [B, max_seq, *feature_dim]
    cat_padded  = pad_sequence(cat_seqs,  batch_first=True, padding_value=0)  # long
    cont_padded = pad_sequence(cont_seqs, batch_first=True, padding_value=0.0) # float

    # 3) build padding mask
    lengths = [seq.size(0) for seq in cat_seqs]  # original seq lengths
    B, max_seq, _ = cat_padded.size()
    device = cat_padded.device
    lengths_tensor = torch.tensor(lengths, device=device)
    idxs = torch.arange(max_seq, device=device).unsqueeze(0).expand(B, -1)
    padding_mask = idxs < lengths_tensor.unsqueeze(1)  # True for real tokens

    # 4) assemble batch dicts
    batch_inputs = {
        "cat":  cat_padded,   # [B, max_seq, C]
        "cont": cont_padded,  # [B, max_seq, F]
    }
    batch_targets = {
        "tgt_cat":  tgt_cat,   # [B, C]
        "tgt_cont": tgt_cont,  # [B, F]
    }

    return batch_inputs, batch_targets, padding_mask



