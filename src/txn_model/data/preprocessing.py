import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from typing import Optional

def preprocessing(
    file: str,
    cols_to_drop: Optional[list[str]] = None,
    date_features: Optional[list[str]] = None,
    cat_features: Optional[list[str]] = None
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
  """
  Takes a csv file path and a list of columns to ignore as
  input and outputs fully processed data ready to be used in a txnDataset.

    - cols_to_drop: list of names of cols to drop
    - date_features: are the names of columns that contain dates to be
      converted to unix time and then standardized.
    - cat_features: a list of categorical features to be encoded
  """
  # to ensure no mutations across calls
  cols_to_drop   = list(cols_to_drop or [])
  date_features  = list(date_features or [])
  cat_features     = list(cat_features or [])

  data = pd.read_csv(file)
  # drop meaningless or sensitive data
  raw = data[data['is_fraud'] == 0].copy() # dont include fraud points
  if cols_to_drop:
    raw.drop(columns=cols_to_drop, inplace=True)
  def add_birthday_features(df):
      # ensure datetimes
      df['dob'] = pd.to_datetime(df['dob'])
      df['txn'] = pd.to_datetime(df['trans_date_trans_time'])

      # age in years
      df['age'] = ((df['txn'] - df['dob'])
                  .dt.total_seconds() / (60 * 60 * 24 * 365.25))

      # birth-month cyclical
      m = df['dob'].dt.month
      df['bmonth_sin'] = np.sin(2 * np.pi * (m-1) / 12)
      df['bmonth_cos'] = np.cos(2 * np.pi * (m-1) / 12)

      # day-of-year for txn and dob
      doy_txn   = df['txn'].dt.dayofyear
      doy_birth = df['dob'].dt.dayofyear

      # days until next birthday (mod 365)
      df['days_to_bday'] = (doy_birth - doy_txn) % 365

      return df

  raw = add_birthday_features(raw)
  if date_features:
    for feat in date_features:
        raw[feat] = pd.to_datetime(raw[feat])
    # compute age in years at transaction time

  if date_features:
      # convert to seconds since epoch
      for feat in date_features:
        raw[feat] = pd.to_datetime(raw[feat])
        raw[f'{feat}_unix'] = raw[feat].astype('int64') // 10**9

      unix_cols = [f'{feat}_unix' for feat in date_features]
      scaler = StandardScaler()
      raw[unix_cols] = scaler.fit_transform(raw[unix_cols])


  # drop the old raw timestamp and rename the cleaned one
  raw.drop(columns=['unix_time', 'trans_date_trans_time', 'dob', 'txn'], errors='ignore', inplace=True)
  raw.rename(columns={'trans_date_trans_time_unix': 'unix_trans_time'}, inplace=True)


  encoders: dict[str, LabelEncoder] = {}
  if cat_features:
    for col in cat_features:
      le = LabelEncoder()
      raw[col] = le.fit_transform(raw[col].astype(str))
      encoders[col] = le

  sorted = raw.sort_values(by=["cc_num", "unix_trans_time"])
  return sorted, encoders
