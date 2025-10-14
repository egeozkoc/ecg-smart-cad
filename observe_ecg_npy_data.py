import os
import sys
import numpy as np


def _pick_default_file() -> str:
    folder = os.path.join(os.path.dirname(__file__), "cad_dataset_preprocessed")
    candidates = [p for p in sorted(os.listdir(folder)) if p.endswith(".npy")]
    if not candidates:
        raise FileNotFoundError("No .npy files found in cad_dataset_preprocessed")
    return os.path.join(folder, candidates[0])


def _to_native(value):
    if isinstance(value, (np.generic,)):
        try:
            return value.item()
        except Exception:
            return value.tolist() if hasattr(value, "tolist") else value
    return value


def _array_stats(arr: np.ndarray) -> str:
    try:
        flat = arr.ravel()
        if flat.size == 0:
            return "empty"
        if np.issubdtype(arr.dtype, np.number):
            return f"min={float(np.min(flat))}, max={float(np.max(flat))}, mean={float(np.mean(flat))}"
        return f"dtype={arr.dtype}, size={arr.size}"
    except Exception as exc:
        return f"unavailable ({exc})"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = set(a for a in sys.argv[1:] if a.startswith("-"))
    show_full = ("--full" in flags) or ("-f" in flags)

    path = args[0] if args else _pick_default_file()
    data = np.load(path, allow_pickle=True)

    print(f"=== File ===")
    print(f"path: {path}")
    print(f"type: {type(data)}")

    if not isinstance(data, np.ndarray):
        print("value:")
        print(repr(data))
        return

    print(f"ndim: {data.ndim}")
    print(f"shape: {data.shape}")
    print(f"dtype: {data.dtype}")

    # If this is a scalar object array holding a dict, summarize its content
    if data.dtype == object and data.shape == ():
        obj = data.item()
        if isinstance(obj, dict):
            keys = list(obj.keys())
            print(f"keys: {keys}")

            # Metadata
            print("\n=== Metadata ===")
            for k in ["id", "filename", "fs", "poor_quality"]:
                if k in obj:
                    print(f"{k}: {_to_native(obj[k])}")
            if "leads" in obj:
                leads = obj["leads"]
                try:
                    lead_list = list(leads)
                    print(f"leads ({len(lead_list)}): {' '.join(map(str, lead_list))}")
                except Exception:
                    print(f"leads: {_to_native(leads)}")
            if "demographics" in obj and isinstance(obj["demographics"], dict):
                demo = obj["demographics"]
                age = demo.get("age")
                sex = demo.get("sex")
                if age is not None:
                    print(f"age: {_to_native(age)}")
                if sex is not None:
                    print(f"sex: {_to_native(sex)}")

            # Waveforms
            if "waveforms" in obj and isinstance(obj["waveforms"], dict):
                print("\n=== Waveforms ===")
                wf = obj["waveforms"]
                for name in ["ecg_10sec_clean", "ecg_median"]:
                    if name in wf and isinstance(wf[name], np.ndarray):
                        arr = wf[name]
                        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, {_array_stats(arr)}")
                # Beats may be a list of arrays
                if "beats" in wf:
                    beats = wf["beats"]
                    try:
                        num_beats = len(beats)
                    except Exception:
                        num_beats = "unknown"
                    print(f"beats: count={num_beats}")
                    try:
                        if num_beats and num_beats != "unknown":
                            first = beats[0]
                            if isinstance(first, np.ndarray):
                                print(f"beats[0]: shape={first.shape}, dtype={first.dtype}, {_array_stats(first)}")
                    except Exception:
                        pass

            # Fiducials
            if "fiducials" in obj and isinstance(obj["fiducials"], dict):
                print("\n=== Fiducials ===")
                fid = obj["fiducials"]
                local = fid.get("local")
                if isinstance(local, dict):
                    print("local:")
                    for k, v in local.items():
                        try:
                            if isinstance(v, np.ndarray):
                                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                            elif isinstance(v, (list, tuple)):
                                print(f"  {k}: len={len(v)}")
                            else:
                                print(f"  {k}: {_to_native(v)}")
                        except Exception:
                            print(f"  {k}: unavailable")
                for k in ["global", "rpeaks", "pacing_spikes", "r_peaks"]:
                    if k in fid:
                        v = fid[k]
                        try:
                            if isinstance(v, np.ndarray):
                                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
                            elif isinstance(v, (list, tuple)):
                                print(f"{k}: len={len(v)}")
                            else:
                                print(f"{k}: {_to_native(v)}")
                        except Exception:
                            print(f"{k}: unavailable")

            # Features
            if "features" in obj and isinstance(obj["features"], dict):
                print("\n=== Features ===")
                feats = obj["features"]
                print(f"count: {len(feats)}")
                sample_keys = sorted(list(feats.keys()))[:15]
                for k in sample_keys:
                    val = _to_native(feats[k])
                    print(f"  {k}: {val}")

                # Save all feature names to features.txt in the repository directory
                try:
                    feature_names = sorted([str(k) for k in feats.keys()])
                    out_path = os.path.join(os.path.dirname(__file__), "features.txt")
                    with open(out_path, "w", encoding="utf-8") as f:
                        for name in feature_names:
                            f.write(f"{name}\n")
                    print(f"saved_feature_names: {len(feature_names)} -> {out_path}")
                except Exception as exc:
                    print(f"could_not_save_feature_names: {exc}")

            if show_full:
                print("\n=== Full Content (raw) ===")
                try:
                    np.set_printoptions(threshold=np.inf, linewidth=200)
                    print(obj)
                except Exception as exc:
                    print(f"could_not_print_full_content: {exc}")
            return

    # Fallback summaries for non-object arrays
    if data.size > 0 and np.issubdtype(data.dtype, np.number):
        print("\n=== Numeric Array Summary ===")
        flat = data.ravel()
        print(f"min: {float(np.min(flat))}")
        print(f"max: {float(np.max(flat))}")
        print(f"mean: {float(np.mean(flat))}")
        print(f"first_values: {flat[:10]}")
    elif data.size > 0 and data.dtype == object:
        print("\n=== Object Array Summary ===")
        try:
            first = data.flat[0]
            print(f"first_element_type: {type(first)}")
            if isinstance(first, dict):
                print(f"first_element_keys: {list(first.keys())}")
        except Exception as exc:
            print(f"could_not_inspect_first_element: {exc}")

    if show_full:
        print("\n=== Full Content (raw) ===")
        try:
            np.set_printoptions(threshold=np.inf, linewidth=200)
            print(data)
        except Exception as exc:
            print(f"could_not_print_full_content: {exc}")


if __name__ == "__main__":
    main()


