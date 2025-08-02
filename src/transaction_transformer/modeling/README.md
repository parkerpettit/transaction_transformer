"""
transaction_model.py
--------------------
Clean, internally-consistent re-implementation of the tabular-sequence model.

Shapes (all tensors use batch-first convention):
    cat          : (B, L, C)   categorical token ids
    cont         : (B, L, F)   continuous features
    field_out    : (B*L, K, D)
    row_repr     : (B, L, M)
    seq_out      : (B, L, M)
    
Note: D is the hidden dimension of the field transformer. M is the hidden dimension of the sequence transformer.
K is the total number of features (C + F). 


"""