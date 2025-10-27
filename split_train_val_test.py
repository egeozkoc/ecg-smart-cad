import os
from typing import Tuple
import csv
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
EXCEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "cad_dataset",
    "20240222_IschemiaAndEKGs_Deidentified.xlsx",
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "cad_dataset_preprocessed")



def _make_label_series(prob_series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(prob_series, errors="coerce")
    # drop 7 later; here map 3/4 -> 1, others -> 0
    return numeric.apply(lambda v: 1 if v in {3, 4} else 0)


def _filter_out_prob7(df: pd.DataFrame) -> pd.DataFrame:
    numeric = pd.to_numeric(df["probIschemia"], errors="coerce")
    mask_keep = numeric.ne(7) & numeric.notna()
    return df.loc[mask_keep].copy()


def _filter_poor_quality(df: pd.DataFrame, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Filter out signals with poor quality flag.
    
    Args:
        df: DataFrame with 'ID' column
        data_dir: Directory containing preprocessed .npy files
        
    Returns:
        DataFrame with poor quality signals removed
    """
    good_quality_ids = []
    poor_quality_ids = []
    missing_ids = []
    
    for id_val in df["ID"]:
        file_path = os.path.join(data_dir, f"{id_val}.npy")
        
        try:
            # Load ECG data
            ecg_data = np.load(file_path, allow_pickle=True).item()
            
            # Check for poor quality flag
            poor_quality = ecg_data.get('poor_quality', False)
            
            if poor_quality:
                poor_quality_ids.append(id_val)
            else:
                good_quality_ids.append(id_val)
        except FileNotFoundError:
            missing_ids.append(id_val)
            continue
        except Exception as e:
            print(f"Warning: Error processing {id_val}: {e}")
            missing_ids.append(id_val)
            continue
    
    # Print statistics
    print(f"\nQuality filtering:")
    print(f"  Total IDs in dataset: {len(df)}")
    print(f"  Good quality: {len(good_quality_ids)}")
    print(f"  Poor quality (removed): {len(poor_quality_ids)}")
    print(f"  Missing files (removed): {len(missing_ids)}")
    
    # Filter dataframe to keep only good quality signals
    filtered_df = df[df["ID"].isin(good_quality_ids)].copy()
    
    return filtered_df


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


def read_ids_from_csv(csv_path: Path) -> list[str]:
    ids: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ecg_id = (row.get("ID") or "").strip()
            if ecg_id:
                ids.append(ecg_id)
    return ids


def copy_split(ids: list[str], source_dir: Path, dest_dir: Path) -> tuple[int, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for ecg_id in ids:
        src_npy = source_dir / f"{ecg_id}.npy"
        src_npz = source_dir / f"{ecg_id}.npz"

        if src_npy.exists():
            shutil.copy2(src_npy, dest_dir / src_npy.name)
            copied += 1
        elif src_npz.exists():
            shutil.copy2(src_npz, dest_dir / src_npz.name)
            copied += 1
        else:
            missing += 1
    return copied, missing

def perform_split():
    project_root = Path(__file__).resolve().parent
    source_dir = project_root / "cad_dataset_preprocessed"

    # Destination directories
    train_dir = source_dir / "train_set"
    val_dir = source_dir / "val_set"
    test_dir = source_dir / "test_set"

    # CSV paths
    train_csv = project_root / "train_set.csv"
    val_csv = project_root / "val_set.csv"
    test_csv = project_root / "test_set.csv"

    # Read IDs
    train_ids = read_ids_from_csv(train_csv)
    val_ids = read_ids_from_csv(val_csv)
    test_ids = read_ids_from_csv(test_csv)

    # Copy files
    train_copied, train_missing = copy_split(train_ids, source_dir, train_dir)
    val_copied, val_missing = copy_split(val_ids, source_dir, val_dir)
    test_copied, test_missing = copy_split(test_ids, source_dir, test_dir)

    # Summary
    print(f"train_set: copied={train_copied} missing={train_missing}")
    print(f"val_set:   copied={val_copied} missing={val_missing}")
    print(f"test_set:  copied={test_copied} missing={test_missing}")


def create_split_data() -> None:
    # Load Excel
    df_excel = pd.read_excel(EXCEL_PATH)

    # Filter out probIschemia == 7 and NaNs, then create binary label
    df_filtered = _filter_out_prob7(df_excel)
    df_filtered["label"] = _make_label_series(df_filtered["probIschemia"])

    # Keep only ID (from target, without .xml) and label
    id_series = df_filtered["target"].astype(str).str.replace(r"\.xml$", "", regex=True)
    df_with_ids = pd.DataFrame({"ID": id_series, "label": df_filtered["label"]})
    
    # Filter out poor quality signals
    df_final = _filter_poor_quality(df_with_ids, data_dir=DATA_DIR)

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
    print("\nFinal dataset split:")
    total = len(df_final)
    _print_stats("Train", train_df, total)
    _print_stats("Val", val_df, total)
    _print_stats("Test", test_df, total)
    
    print(f"\nCSV files saved:")
    print(f"  {out_train}")
    print(f"  {out_val}")
    print(f"  {out_test}")


if __name__ == "__main__":
    create_split_data()
    perform_split()


