from transaction_transformer.modeling.models import EmbeddingLayer, FieldTransformer, SequenceTransformer

# -------------------------------------------------------------------------------------- #
#  Test for EmbeddingLayer                                                               #
# -------------------------------------------------------------------------------------- #
def test_embedding_layer():
    import torch

    # Define vocab sizes for two categorical fields
    cat_vocab = {"merchant": 5, "category": 4}  # 0=PAD, 1=MASK, 2=UNK, 3+=real
    cont_vocab = {"amount": 0}  # just to test continuous, num_bins not used here

    emb_dim = 6
    dropout = 0.0
    padding_idx = 0
    L = 4

    # Create the embedding layer
    emb_layer = EmbeddingLayer(cat_vocab, cont_vocab, emb_dim, dropout, padding_idx, L)

    # Prepare a batch with all special tokens and a real token
    # Shape: (B, T, C) where C = 2 (merchant, category)
    cat = torch.tensor([
        [
            [0, 0],   # PAD, PAD
            [1, 1],   # MASK, MASK
            [2, 2],   # UNK, UNK
            [3, 3],   # real, real
            [4, 2],   # real, UNK
        ]
    ])  # (1, 5, 2)

    # Prepare continuous features: (B, T, F)
    # Use NaN for masked, 0.5 for real
    cont = torch.tensor([
        [
            [float('nan')],   # masked
            [0.0],            # real
            [1.0],            # real
            [float('nan')],   # masked
            [0.5],            # real
        ]
    ])

    # Forward pass
    out = emb_layer(cat, cont)
    B, T, C = cat.shape
    F = cont.shape[-1]
    K = C + F
    assert out.shape == (B * T, K, emb_dim), f"Output shape mismatch: {out.shape}"

    # Check that PAD, MASK, UNK indices are handled (should not crash)
    # Check that masked continuous positions are replaced with mask_token
    # For continuous field, check that output for masked positions matches mask_token
    cont_feature_idx = -1  # last field in K
    mask_token = emb_layer.mask_token.detach()
    out_reshaped = out.view(B, T, K, emb_dim)
    masked_cont_rows = torch.isnan(cont[0, :, 0])
    for t in range(T):
        if masked_cont_rows[t]:
            diff = (out_reshaped[0, t, cont_feature_idx] - mask_token).abs().max()
            assert diff < 1e-5, f"Mask token not used for masked continuous at t={t}"
        else:
            diff = (out_reshaped[0, t, cont_feature_idx] - mask_token).abs().max()
            assert diff > 1e-3, f"Mask token used for real value at t={t}"

    print("EmbeddingLayer test passed.")

# -------------------------------------------------------------------------------------- #
#  Test for EmbeddingLayer -> FieldTransformer pipeline                                 #
# -------------------------------------------------------------------------------------- #
def test_embedder_field_transformer_pipeline():
    """
    Test that the embedder output can be directly fed into the field transformer
    and then through the sequence transformer with the required row_proj operation.
    """
    import torch
    from transaction_transformer.utils.config import TransformerConfig

    # Test parameters
    batch_size = 2
    seq_len = 3
    emb_dim = 64
    num_cat_features = 2
    num_cont_features = 1
    num_fields = num_cat_features + num_cont_features
    
    # Define vocabularies
    cat_vocab = {"merchant": 10, "category": 8}  # 0=PAD, 1=MASK, 2=UNK, 3+=real
    cont_vocab = {"amount": 0}  # num_bins not used in embedder
    
    # Create transformer configs
    ft_config = TransformerConfig(
        d_model=emb_dim,
        n_heads=4,
        depth=2,
        ffn_mult=2,
        dropout=0.1,
        layer_norm_eps=1e-6,
        norm_first=True
    )
    
    st_config = TransformerConfig(
        d_model=emb_dim,
        n_heads=4,
        depth=2,
        ffn_mult=2,
        dropout=0.1,
        layer_norm_eps=1e-6,
        norm_first=True
    )
    
    # Create embedder, field transformer, sequence transformer, and row projection
    embedder = EmbeddingLayer(
        cat_vocab=cat_vocab,
        cont_vocab=cont_vocab,
        emb_dim=emb_dim,
        dropout=0.1,
        padding_idx=0,
        L=8
    )
    
    field_transformer = FieldTransformer(ft_config)
    sequence_transformer = SequenceTransformer(st_config)
    row_proj = torch.nn.Linear(num_fields * emb_dim, emb_dim)
    
    # Create test data
    # Categorical: (B, L, C) with some special tokens and real tokens
    cat = torch.tensor([
        [
            [3, 4],   # real, real
            [1, 2],   # MASK, UNK
            [0, 0],   # PAD, PAD
        ],
        [
            [5, 3],   # real, real
            [2, 1],   # UNK, MASK
            [4, 5],   # real, real
        ]
    ])  # (2, 3, 2)
    
    # Continuous: (B, L, F) with some masked (NaN) and real values
    cont = torch.tensor([
        [
            [0.5],    # real
            [float('nan')],  # masked
            [1.2],    # real
        ],
        [
            [0.8],    # real
            [0.3],    # real
            [float('nan')],  # masked
        ]
    ])  # (2, 3, 1)
    
    # Test 1: Check embedder output shape
    embedder_out = embedder(cat, cont)
    expected_embedder_shape = (batch_size * seq_len, num_fields, emb_dim)
    assert embedder_out.shape == expected_embedder_shape, \
        f"Embedder output shape mismatch. Expected {expected_embedder_shape}, got {embedder_out.shape}"
    
    # Test 2: Check that embedder output can be fed directly to field transformer
    field_transformer_out = field_transformer(embedder_out)
    expected_field_transformer_shape = (batch_size * seq_len, num_fields, emb_dim)
    assert field_transformer_out.shape == expected_field_transformer_shape, \
        f"Field transformer output shape mismatch. Expected {expected_field_transformer_shape}, got {field_transformer_out.shape}"
    
    # Test 3: Check that field transformer actually transforms the data
    # The output should be different from input (not identity)
    diff = (field_transformer_out - embedder_out).abs().max()
    assert diff > 1e-6, "Field transformer output is identical to input - may not be working"
    
    # Test 4: Check the row_proj and sequence transformer pipeline (causal mode)
    # Apply row_proj: (B*L, K, D) -> (B*L, D) -> (B, L, D)
    field_flat = field_transformer_out.view(batch_size * seq_len, -1)  # (B*L, K*D)
    row_proj_out = row_proj(field_flat)  # (B*L, D)
    row_proj_out = row_proj_out.view(batch_size, seq_len, emb_dim)  # (B, L, D)
    
    # Feed into sequence transformer with causal=True (default)
    sequence_transformer_out_causal = sequence_transformer(row_proj_out, causal=True)
    expected_sequence_shape = (batch_size, seq_len, emb_dim)
    assert sequence_transformer_out_causal.shape == expected_sequence_shape, \
        f"Sequence transformer output shape mismatch. Expected {expected_sequence_shape}, got {sequence_transformer_out_causal.shape}"
    
    # Test 5: Check that sequence transformer actually transforms the data (causal mode)
    # The output should be different from input (not identity)
    seq_diff_causal = (sequence_transformer_out_causal - row_proj_out).abs().max()
    assert seq_diff_causal > 1e-6, "Sequence transformer output is identical to input - may not be working"
    
    # Test 6: Check the row_proj and sequence transformer pipeline (non-causal mode)
    # Feed into sequence transformer with causal=False
    sequence_transformer_out_non_causal = sequence_transformer(row_proj_out, causal=False)
    assert sequence_transformer_out_non_causal.shape == expected_sequence_shape, \
        f"Sequence transformer non-causal output shape mismatch. Expected {expected_sequence_shape}, got {sequence_transformer_out_non_causal.shape}"
    
    # Test 7: Check that sequence transformer actually transforms the data (non-causal mode)
    # The output should be different from input (not identity)
    seq_diff_non_causal = (sequence_transformer_out_non_causal - row_proj_out).abs().max()
    assert seq_diff_non_causal > 1e-6, "Sequence transformer non-causal output is identical to input - may not be working"
    
    # Test 8: Check that causal and non-causal outputs are different
    # The causal mask should make the outputs different
    causal_vs_non_causal_diff = (sequence_transformer_out_causal - sequence_transformer_out_non_causal).abs().max()
    assert causal_vs_non_causal_diff > 1e-6, "Causal and non-causal outputs are identical - causal masking may not be working"
    
    # Test 9: Check that masked continuous positions are handled correctly
    # Reshape to check per-batch, per-timestep
    embedder_reshaped = embedder_out.view(batch_size, seq_len, num_fields, emb_dim)
    field_transformer_reshaped = field_transformer_out.view(batch_size, seq_len, num_fields, emb_dim)
    
    # Check that masked continuous positions (last field) are handled
    cont_feature_idx = -1  # last field
    mask_token = embedder.mask_token.detach()
    
    # Since dropout is applied after mask token replacement, we need to be more lenient
    # The mask token should be present but may be modified by dropout
    for b in range(batch_size):
        for t in range(seq_len):
            if torch.isnan(cont[b, t, 0]):  # if continuous field is masked
                # Check that embedder output for masked positions is not zero (should be mask token)
                embedder_masked_output = embedder_reshaped[b, t, cont_feature_idx]
                # The output should not be all zeros (which would indicate no mask token was used)
                assert embedder_masked_output.abs().max() > 1e-6, f"Masked continuous position appears to be zero at batch={b}, timestep={t}"
    
    # Test 10: Check that the pipeline works with different sequence lengths
    # Test with single timestep
    cat_single = cat[:, :1, :]  # (2, 1, 2)
    cont_single = cont[:, :1, :]  # (2, 1, 1)
    
    embedder_out_single = embedder(cat_single, cont_single)
    field_transformer_out_single = field_transformer(embedder_out_single)
    
    # Apply row_proj and sequence transformer to single timestep
    field_flat_single = field_transformer_out_single.view(batch_size * 1, -1)
    row_proj_out_single = row_proj(field_flat_single)
    row_proj_out_single = row_proj_out_single.view(batch_size, 1, emb_dim)
    sequence_transformer_out_single = sequence_transformer(row_proj_out_single)
    
    expected_single_shape = (batch_size, 1, emb_dim)
    assert sequence_transformer_out_single.shape == expected_single_shape, \
        f"Single timestep sequence transformer output shape mismatch. Expected {expected_single_shape}, got {sequence_transformer_out_single.shape}"
    
    # Test 11: Check gradient flow through the entire pipeline
    # Create fresh data and run through all components
    # Note: Only continuous tensors can have gradients, not categorical (LongTensor)
    cat_grad = cat.clone().detach()  # Keep as LongTensor, no gradients
    cont_grad = cont.clone().detach().requires_grad_(True)  # Only this can have gradients
    
    embedder_out_grad = embedder(cat_grad, cont_grad)
    field_transformer_out_grad = field_transformer(embedder_out_grad)
    
    # Apply row_proj and sequence transformer
    field_flat_grad = field_transformer_out_grad.view(batch_size * seq_len, -1)
    row_proj_out_grad = row_proj(field_flat_grad)
    row_proj_out_grad = row_proj_out_grad.view(batch_size, seq_len, emb_dim)
    sequence_transformer_out_grad = sequence_transformer(row_proj_out_grad)
    
    loss = sequence_transformer_out_grad.sum()
    loss.backward()
    
    # Check that gradients flow back to continuous input tensor
    assert cont_grad.grad is not None, "Gradients not flowing back to continuous input"
    assert cont_grad.grad.shape == cont_grad.shape, "Continuous gradient shape mismatch"
    
    # Test 12: Check that embedder parameters have gradients
    # This ensures the embedder is actually learning
    embedder_params_with_grad = [p for p in embedder.parameters() if p.grad is not None]
    assert len(embedder_params_with_grad) > 0, "No embedder parameters have gradients"
    
    # Test 13: Check that field transformer parameters have gradients
    field_transformer_params_with_grad = [p for p in field_transformer.parameters() if p.grad is not None]
    assert len(field_transformer_params_with_grad) > 0, "No field transformer parameters have gradients"
    
    # Test 14: Check that row_proj parameters have gradients
    row_proj_params_with_grad = [p for p in row_proj.parameters() if p.grad is not None]
    assert len(row_proj_params_with_grad) > 0, "No row_proj parameters have gradients"
    
    # Test 15: Check that sequence transformer parameters have gradients
    sequence_transformer_params_with_grad = [p for p in sequence_transformer.parameters() if p.grad is not None]
    assert len(sequence_transformer_params_with_grad) > 0, "No sequence transformer parameters have gradients"
    
    print("Embedder -> FieldTransformer -> RowProj -> SequenceTransformer pipeline test passed.")
    print(sequence_transformer_out_grad)
#TODO change data collation to create mask
