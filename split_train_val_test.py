import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
EXCEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "cad_dataset",
    "20240222_IschemiaAndEKGs_Deidentified.xlsx",
)



def _make_label_series(prob_series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(prob_series, errors="coerce")
    # drop 7 later; here map 3/4 -> 1, others -> 0
    return numeric.apply(lambda v: 1 if v in {3, 4} else 0)


def _filter_out_prob7(df: pd.DataFrame) -> pd.DataFrame:
    numeric = pd.to_numeric(df["probIschemia"], errors="coerce")
    mask_keep = numeric.ne(7) & numeric.notna()
    return df.loc[mask_keep].copy()


def _stratified_split(
    df: pd.DataFrame, label_col: str, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[label_col]
    # If stratification is not feasible (e.g., single class), fall back to random split
    can_stratify = y.nunique(dropna=True) > 1 and all(
        y.value_counts() >= 3
    )

    if can_stratify:
        train_df, temp_df = train_test_split(
            df, test_size=0.2, stratify=y, random_state=seed, shuffle=True
        )
        y_temp = temp_df[label_col]
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=y_temp,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_df, temp_df = train_test_split(
            df, test_size=0.2, random_state=seed, shuffle=True
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=seed, shuffle=True
        )

    return train_df, val_df, test_df


def _print_stats(name: str, part_df: pd.DataFrame, total: int) -> None:
    n = len(part_df)
    ratio_of_total = n / total if total else 0.0
    positive_ratio = (
        part_df["label"].mean() if n > 0 else 0.0
    )  # mean of 0/1 equals positive ratio
    pos_count = int(part_df["label"].sum()) if n > 0 else 0
    neg_count = int(n - pos_count)
    print(
        f"{name}: count={n}, fraction_of_total={ratio_of_total:.4f}, positive_ratio={positive_ratio:.4f}, label0={neg_count}, label1={pos_count}"
    )


def main() -> None:
    # Load Excel
    df_excel = pd.read_excel(EXCEL_PATH)

    # Filter out probIschemia == 7 and NaNs, then create binary label
    df_filtered = _filter_out_prob7(df_excel)
    df_filtered["label"] = _make_label_series(df_filtered["probIschemia"])

    # Keep only ID (from target, without .xml) and label
    id_series = df_filtered["target"].astype(str).str.replace(r"\.xml$", "", regex=True)
    df_final = pd.DataFrame({"ID": id_series, "label": df_filtered["label"]})

    # Perform stratified 80/10/10 split
    train_df, val_df, test_df = _stratified_split(df_final, label_col="label", seed=SEED)

    # Save CSVs
    out_train = os.path.join(os.path.dirname(__file__), "train_set.csv")
    out_val = os.path.join(os.path.dirname(__file__), "val_set.csv")
    out_test = os.path.join(os.path.dirname(__file__), "test_set.csv")
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    # Print stats
    total = len(df_final)
    _print_stats("Train", train_df, total)
    _print_stats("Val", val_df, total)
    _print_stats("Test", test_df, total)


if __name__ == "__main__":
    main()


