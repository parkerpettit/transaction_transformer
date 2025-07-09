from data.dataset import TxnDataset, collate_fn, TxnCtxDataset, collate_fn_ctx
from config import ModelConfig, FieldTransformerConfig, SequenceTransformerConfig, LSTMConfig
from data.preprocessing import preprocess, preprocess_for_latents_full
from train import train
from torch.utils.data import DataLoader
import joblib

# orig_cont = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long'] #'bmonth_cos', "days_to_bday",
# orig_cat = ["cc_num", "merchant", "category", "gender", "city", "state", "job"]
# date_feats_to_expand = ["trans_date_trans_time"]
# train_df, val_df, test_df, encoders, cat_features, cont_features, scaler = preprocess("credit_card_transactions.csv",
#                      cat_features=orig_cat.copy(),
#                      cont_features=orig_cont.copy(),
#                      date_feats_to_expand=date_feats_to_expand
#                      )
cache_path = "/content/drive/MyDrive/summer_urop_25/datasets/processed_data.joblib"

(
    train_df,
    val_df,
    test_df,
    encoders,
    cat_features,
    cont_features,
    scaler
) = joblib.load(cache_path)

cat_vocab_sizes = {col: len(enc["inv"]) for col, enc in encoders.items()}


train_ds = TxnDataset(
    df=train_df,
    group_by="User",
    cat_features=cat_features,
    cont_features=cont_features,
    window_size=10,
    stride=5
)

val_ds = TxnDataset(
    df=val_df,
    group_by="User",
    cat_features=cat_features,
    cont_features=cont_features,
    window_size=10,
    stride=5
)

test_ds = TxnDataset(
    df=test_df,
    group_by="User",
    cat_features=cat_features,
    cont_features=cont_features,
    window_size=10,
    stride=5
)



train_loader = DataLoader(train_ds, shuffle=True,
                          batch_size=32, num_workers=4,
                          collate_fn=collate_fn, pin_memory=True)

val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                          num_workers=4, collate_fn=collate_fn, pin_memory=True)

test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                          num_workers=4, collate_fn=collate_fn, pin_memory=True)


# ------------------------------------------------------------------
EMB_DIM    = 48       # “d”  – per-field embedding size
ROW_DIM    = 256     # “m” – row vector after W_h, before sequence transformer
HEADS_F    = 4        # field transformer heads (72 / 8 = 9 dim/head)
HEADS_S    = 4       # sequence transformer heads (1080 / 12 = 90 dim/head)
DEPTH_F    = 1        # layers in FieldTransformer (paper: 1)
DEPTH_S    = 4       # layers in SequenceTransformer (paper: 12)
FFN_MULT   = 2        # feed-forward dim = 4 × d_model (paper default)
DROPOUT    = 0.10
LN_EPS     = 1e-6     # paper uses 1e-6
LSTM_HID   = ROW_DIM     # hidden size of LSTM
LSTM_NUM_LAYERS = 2
NUM_CLASSES = 2


# ------------------------------------------------------------------

#  Intra-row (Field) transformer config
field_cfg = FieldTransformerConfig(
    d_model        = EMB_DIM,
    n_heads        = HEADS_F,
    depth          = DEPTH_F,
    ffn_mult       = FFN_MULT,
    dropout        = DROPOUT,
    layer_norm_eps = LN_EPS,
    norm_first     = True          # pre-norm like UniTTab
)

#   Inter-row (Sequence) transformer config
sequence_cfg = SequenceTransformerConfig(
    d_model        = ROW_DIM,
    n_heads        = HEADS_S,
    depth          = DEPTH_S,
    ffn_mult       = FFN_MULT,
    dropout        = DROPOUT,
    layer_norm_eps = LN_EPS,
    norm_first     = True
)

lstm_cfg = LSTMConfig(
  hidden_size=LSTM_HID,
  num_layers=LSTM_NUM_LAYERS,
  num_classes=NUM_CLASSES,
  dropout=DROPOUT
)

#   Assemble the global model config
config = ModelConfig(
    cat_vocab_sizes = cat_vocab_sizes,   # dict[str,int]
    cont_features   = cont_features,     # list[str]
    emb_dim         = EMB_DIM,           # 64
    dropout         = DROPOUT,           # 0.1
    padding_idx     = 0,
    field_transformer    = field_cfg,
    sequence_transformer = sequence_cfg,
    lstm = lstm_cfg
)

train(config=config,
      cat_vocab_mapping=cat_vocab_sizes,
      cat_features=cat_features,
      cont_features=cont_features,
      train_loader=train_loader,
      val_loader=val_loader
      )

# train_df, val_df, test_df, cat_feats, cont_feats = preprocess_for_latents_full(
#     file                = "credit_card_transactions.csv",
#     cat_features        = orig_cat,
#     cont_features       = orig_cont,
#     encoders            = encoders,    # from training checkpoint
#     scaler              = scaler,      # ditto
#     date_feats_to_expand= ["trans_date_trans_time"],
#     date_parts          = {"trans_date_trans_time": ["year","month", "day", "hour"]}
# )



