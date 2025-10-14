import csv
import shutil
from pathlib import Path


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


def main() -> None:
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


if __name__ == "__main__":
    main()


