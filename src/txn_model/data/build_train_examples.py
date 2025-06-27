def build_train_examples(
    df,
    group_key: str,
    cat_fields: list[str],
    cont_fields: list[str],
) -> list[dict]:
    """
    Convert a sorted transaction DataFrame into a list of example dicts for txnDataset.

    Args:
        df: pandas DataFrame containing all transactions, already with categorical
            fields mapped to integer IDs and continuous fields as floats.
        group_key: column name to group by (e.g. 'cc_num').
        cat_fields: list of base categorical field names (no prefix).
        cont_fields: list of base continuous field names (no prefix).

    Returns:
        List of dicts, each with keys:
            'cat_<field>'   -> List[int] history sequence
            'cont_<field>'  -> List[float] history sequence
            'target_cat_<field>'  -> int scalar for next-row
            'target_cont_<field>' -> float scalar for next-row
    """
    examples: list[dict] = []

    for _, user_df in df.groupby(group_key):
      rows = user_df.to_dict("records")
      L = len(rows)
      if L < 2:
        continue # need at least one transaction + target
      for t in range(L - 1):
        context = rows[:t + 1]
        next = rows[t+1]
        ex: dict = {}
        # build context sequences
        for field in cat_fields:
          ex[f'cat_{field}'] = [row[field] for row in context]
        for field in cont_fields:
          ex[f'cont_{field}'] = [row[field] for row in context]
        # build target sequences
        for field in cat_fields:
          ex[f'tgt_cat_{field}'] = next[field]
        for field in cont_fields:
          ex[f'tgt_cont_{field}'] = next[field]
        examples.append(ex)
      return examples