import pandas as pd
from pathlib import Path
from transaction_transformer.data.preprocessing import preprocess, FieldSchema, get_encoders, get_scaler, build_quantile_binner, normalize, encode_df 



if __name__ == "__main__":
    cat_features = [
        "User",
        "Card",
        "Use Chip",
        "Merchant Name",
        "Merchant City",
        "Merchant State",
        "Zip",
        "MCC",
        "Errors?",
        "Year",
        "Month",
        "Day",
        "Hour",
    ]

    cont_features = ["Amount"]

    print("Preprocessing starting")
    full_train_df, full_val_df, full_test_df = preprocess(Path("card_transaction.v1.csv"))
    print("Preprocessing done")
    print("Getting scaler")

    scaler = get_scaler(full_train_df)
    binner = build_quantile_binner(full_train_df["Amount"]) # type: ignore
    # get encoders from entire dataset
    print("Getting encoders")

    cat_encoders = get_encoders(pd.concat((full_train_df, full_val_df, full_test_df), axis=0), cat_features=cat_features)

    print("Applying encoders and scalers to dfs")


    schema = FieldSchema(
        cat_features=cat_features,
        cont_features=cont_features,
        cat_encoders={c: cat_encoders[c] for c in cat_features},
        cont_binners={f: binner for f in cont_features},
        time_cat=["Year", "Month", "Day", "Hour"],  
        scaler=scaler,
    )

    full_train_df, full_val_df, full_test_df = [
        normalize(encode_df(df.copy(), schema.cat_encoders, schema.cat_features), scaler, schema.cont_features)
        for df in (full_train_df, full_val_df, full_test_df)
    ]
    print("Removing fraud examples from legit dfs")
    legit_train, legit_val, legit_test = [
        df[df["is_fraud"] == 0].copy().reset_index(drop=True)
        for df in (full_train_df, full_val_df, full_test_df)
    ]

    assert len(full_train_df) > len(legit_train)
    assert len(full_val_df) > len(legit_val)
    assert len(full_test_df) > len(legit_test)

    assert len(legit_train) == len(full_train_df) - len(full_train_df[full_train_df["is_fraud"] == 1])
    assert len(legit_val) == len(full_val_df) - len(full_val_df[full_val_df["is_fraud"] == 1])
    assert len(legit_test) == len(full_test_df) - len(full_test_df[full_test_df["is_fraud"] == 1])

    assert legit_train["is_fraud"].sum() == 0
    assert legit_val["is_fraud"].sum() == 0
    assert legit_test["is_fraud"].sum() == 0

    print("Saving")
    import torch
    torch.save((full_train_df, full_val_df, full_test_df, schema), "full_processed.pt")
    torch.save((legit_train, legit_val, legit_test, schema), "legit_processed.pt")



