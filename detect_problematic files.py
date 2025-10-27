import numpy as np
import pandas as pd
from scipy import signal
import os
import warnings
from visualize_ecg_signal import visualize_ecg_grid

def detect_problematic_signals(data_dir='cad_dataset_preprocessed/'):
    """
    Detect ECG signals that cause division by zero errors during preprocessing.
    Replicates the exact preprocessing from train_models.py to find problematic signals.
    """
    
    # Load the CSV files with train/val/test splits
    train_df = pd.read_csv('train_set.csv')
    val_df = pd.read_csv('val_set.csv')
    test_df = pd.read_csv('test_set.csv')
    
    # Combine all IDs
    all_ids = train_df['ID'].to_list() + val_df['ID'].to_list() + test_df['ID'].to_list()
    all_splits = ['train'] * len(train_df) + ['val'] * len(val_df) + ['test'] * len(test_df)
    
    print(f"Checking {len(all_ids)} ECG files for division by zero issues and quality flags...")
    print("=" * 80)
    
    problematic_files = []
    missing_files = []
    low_quality_files = []
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    for idx, (file_id, split) in enumerate(zip(all_ids, all_splits)):
        file_path = os.path.join(data_dir, file_id + '.npy')
        
        try:
            # Load ECG data
            ecg_data = np.load(file_path, allow_pickle=True).item()
            ecg = ecg_data['waveforms']['ecg_median']
            
            # Check for low quality flag
            poor_quality = ecg_data.get('poor_quality', False)
            if poor_quality:
                low_quality_files.append({
                    'file_id': file_id,
                    'split': split,
                    'poor_quality': poor_quality
                })
            
            # Apply the same preprocessing as in train_models.py (ORIGINAL BUGGY VERSION)
            ecg_cropped = ecg[:, 150:-50]
            ecg_resampled = signal.resample(ecg_cropped, 200, axis=1)
            
            # Check for zero max values (the problematic case)
            # This replicates the ORIGINAL buggy code that caused division by zero
            max_val = np.max(np.abs(ecg_resampled), axis=1)
            
            # The original code checked: if np.sum(max_val) > 0: ecg / max_val
            # This would cause division by zero if ANY individual lead had max_val[i] == 0
            # even if the sum was > 0
            zero_leads = np.where(max_val == 0)[0]
            
            # Also check if this would have passed the buggy condition
            sum_check = np.sum(max_val) > 0
            
            # Actually try the division to see if it produces a warning
            caught_warning = False
            if sum_check:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _ = ecg_resampled / max_val[:, None]
                    if len(w) > 0:
                        for warning in w:
                            if "invalid value" in str(warning.message) or "divide" in str(warning.message):
                                caught_warning = True
                                break
            
            # Only report if there are zero leads AND the sum check would have passed
            # (meaning the buggy code would have tried to divide by zero)
            if (len(zero_leads) > 0 and sum_check) or caught_warning:
                zero_lead_names = [lead_names[i] for i in zero_leads]
                problematic_files.append({
                    'file_id': file_id,
                    'file_path': file_path,
                    'split': split,
                    'zero_leads': zero_lead_names,
                    'zero_lead_indices': zero_leads,
                    'num_zero_leads': len(zero_leads),
                    'would_cause_warning': True
                })
                
                print(f"⚠️  [{split.upper()}] {file_id}.npy")
                print(f"  └─ Leads with all zeros (would cause div by 0): {', '.join(zero_lead_names)}")
                print(f"  └─ Max values: {max_val}")
                print()
        
        except FileNotFoundError:
            missing_files.append(file_id)
            continue
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            continue
    
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total files in CSV: {len(all_ids)}")
    print(f"  Files missing from directory: {len(missing_files)}")
    print(f"  Files successfully processed: {len(all_ids) - len(missing_files)}")
    print(f"  Files with division by zero issues: {len(problematic_files)}")
    print(f"  Files with low quality flag: {len(low_quality_files)}")
    
    # Print low quality breakdown by split
    if len(low_quality_files) > 0:
        train_lq = sum(1 for f in low_quality_files if f['split'] == 'train')
        val_lq = sum(1 for f in low_quality_files if f['split'] == 'val')
        test_lq = sum(1 for f in low_quality_files if f['split'] == 'test')
        
        print(f"    - Train set: {train_lq}")
        print(f"    - Val set: {val_lq}")
        print(f"    - Test set: {test_lq}")
    
    if len(problematic_files) > 0:
        # Count by split
        train_count = sum(1 for f in problematic_files if f['split'] == 'train')
        val_count = sum(1 for f in problematic_files if f['split'] == 'val')
        test_count = sum(1 for f in problematic_files if f['split'] == 'test')
        
        print(f"    - Train set: {train_count}")
        print(f"    - Val set: {val_count}")
        print(f"    - Test set: {test_count}")
        
        # Count total zero leads
        total_zero_leads = sum(f['num_zero_leads'] for f in problematic_files)
        print(f"  Total leads affected: {total_zero_leads}")
        
        # Most common zero leads
        from collections import Counter
        all_zero_leads = []
        for f in problematic_files:
            all_zero_leads.extend(f['zero_leads'])
        lead_counts = Counter(all_zero_leads)
        
        print(f"\n  Most commonly affected leads:")
        for lead, count in lead_counts.most_common():
            print(f"    - {lead}: {count} times")
    
    return problematic_files, low_quality_files


def visualize_problematic_signals(problematic_files, max_to_show=5):
    """
    Visualize the problematic ECG signals.
    
    Parameters:
    -----------
    problematic_files : list
        List of dictionaries containing info about problematic files
    max_to_show : int
        Maximum number of signals to visualize
    """
    import matplotlib.pyplot as plt
    
    if len(problematic_files) == 0:
        print("\nNo problematic signals to visualize!")
        return
    
    print(f"\n{'='*80}")
    print(f"Visualizing up to {max_to_show} problematic signals...")
    print(f"{'='*80}\n")
    
    num_to_show = min(max_to_show, len(problematic_files))
    
    for i in range(num_to_show):
        file_info = problematic_files[i]
        file_path = file_info['file_path']
        file_id = file_info['file_id']
        split = file_info['split']
        zero_leads = file_info['zero_leads']
        
        print(f"Visualizing {i+1}/{num_to_show}: {file_id}.npy ({split} set)")
        print(f"  Problematic leads: {', '.join(zero_leads)}")
        
        # Create visualizations
        fig1, _ = visualize_ecg_grid(file_path, apply_preprocessing=False)
        fig2, _ = visualize_ecg_grid(file_path, apply_preprocessing=True)
        
        plt.tight_layout()
    
    print(f"\nDisplaying {num_to_show} figure pairs (original + preprocessed)...")
    plt.show()


def save_report(problematic_files, low_quality_files, output_file='division_by_zero_report.txt'):
    """
    Save a detailed report of problematic signals and low quality files to a text file.
    """
    with open(output_file, 'w') as f:
        f.write("ECG Signal Quality Report\n")
        f.write("=" * 80 + "\n\n")
        
        # Division by zero section
        f.write("DIVISION BY ZERO ERRORS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total problematic files: {len(problematic_files)}\n\n")
        
        if len(problematic_files) > 0:
            f.write("Detailed List:\n")
            for i, file_info in enumerate(problematic_files, 1):
                f.write(f"{i}. {file_info['file_id']}.npy\n")
                f.write(f"   Split: {file_info['split']}\n")
                f.write(f"   Problematic leads: {', '.join(file_info['zero_leads'])}\n")
                f.write(f"   Number of zero leads: {file_info['num_zero_leads']}\n")
                f.write("\n")
        else:
            f.write("No division by zero errors found.\n\n")
        
        # Low quality section
        f.write("\n" + "=" * 80 + "\n")
        f.write("LOW QUALITY ECG FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total low quality files: {len(low_quality_files)}\n\n")
        
        if len(low_quality_files) > 0:
            # Group by split
            train_lq = [f for f in low_quality_files if f['split'] == 'train']
            val_lq = [f for f in low_quality_files if f['split'] == 'val']
            test_lq = [f for f in low_quality_files if f['split'] == 'test']
            
            f.write(f"By split:\n")
            f.write(f"  Train: {len(train_lq)}\n")
            f.write(f"  Val: {len(val_lq)}\n")
            f.write(f"  Test: {len(test_lq)}\n\n")
            
            f.write("Detailed List:\n")
            for i, file_info in enumerate(low_quality_files, 1):
                f.write(f"{i}. {file_info['file_id']}.npy (split: {file_info['split']})\n")
        else:
            f.write("No low quality files found.\n")
    
    print(f"\nDetailed report saved to: {output_file}")


if __name__ == '__main__':
    # Detect problematic signals and low quality files
    visualize_flag = False
    problematic_files, low_quality_files = detect_problematic_signals()
    
    # Save report
    if len(problematic_files) > 0 or len(low_quality_files) > 0:
        save_report(problematic_files, low_quality_files)
    
    # Visualize problematic signals if any
    if len(problematic_files) > 0:
        # Ask if user wants to visualize
        print("\n" + "=" * 80)
        num_to_show = min(5, len(problematic_files))
        print(f"\nWould you like to visualize the problematic signals?")
        print(f"(Will show {num_to_show} out of {len(problematic_files)} problematic files)")
        
        # For non-interactive execution, automatically visualize
        if visualize_flag:
            visualize_problematic_signals(problematic_files, max_to_show=5)
    else:
        print("\nNo division by zero issues found! All signals are clean.")

