import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def visualize_ecg(file_path, apply_preprocessing=False):
    """
    Visualize a 12-lead ECG signal
    
    Parameters:
    -----------
    file_path : str
        Path to the .npy file containing ECG data
    apply_preprocessing : bool
        If True, apply the same preprocessing as in training (crop and resample)
    """
    # Load ECG data
    ecg_data = np.load(file_path, allow_pickle=True).item()
    ecg = ecg_data['waveforms']['ecg_median']
    
    # Extract filename from path
    filename = os.path.basename(file_path)
    
    # Standard 12-lead ECG names
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Apply preprocessing if requested
    if apply_preprocessing:
        ecg_processed = ecg[:, 150:-50]
        ecg_processed = signal.resample(ecg_processed, 200, axis=1)
        # Normalize each lead (same as training)
        max_val = np.max(np.abs(ecg_processed), axis=1, keepdims=True)
        max_val = np.where(max_val == 0, 1, max_val)  # Replace zeros with 1
        ecg_processed = ecg_processed / max_val
        sample_axis = np.arange(ecg_processed.shape[1])
        ecg_to_plot = ecg_processed
        title_suffix = " (Preprocessed)"
    else:
        sample_axis = np.arange(ecg.shape[1])
        ecg_to_plot = ecg
        title_suffix = " (Original)"
    
    # Create figure with subplots for each lead
    fig, axes = plt.subplots(12, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'12-Lead ECG Signal: {filename}{title_suffix}', fontsize=16, fontweight='bold')
    
    # Plot each lead
    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        # Check if lead has all zeros
        if np.all(ecg_to_plot[i] == 0):
            ax.plot(sample_axis, ecg_to_plot[i], 'r-', linewidth=1.5, alpha=0.7)
        else:
            ax.plot(sample_axis, ecg_to_plot[i], 'b-', linewidth=1.5)
        
        # Styling
        ax.set_ylabel(lead_name, fontsize=12, fontweight='bold', rotation=0, labelpad=20)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Show statistics
        if not np.all(ecg_to_plot[i] == 0):
            max_val = np.max(np.abs(ecg_to_plot[i]))
            ax.text(0.98, 0.95, f'{max_val:.2f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set common x-label
    axes[-1].set_xlabel('Samples', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig, axes


def visualize_ecg_grid(file_path, apply_preprocessing=False):
    """
    Visualize ECG in a more traditional 3x4 grid layout (like printed ECG)
    
    Parameters:
    -----------
    file_path : str
        Path to the .npy file containing ECG data
    apply_preprocessing : bool
        If True, apply the same preprocessing as in training
    """
    # Load ECG data
    ecg_data = np.load(file_path, allow_pickle=True).item()
    ecg = ecg_data['waveforms']['ecg_median']
    
    # Extract filename from path
    filename = os.path.basename(file_path)
    
    # Standard 12-lead ECG names in typical grid arrangement
    lead_names = [
        ['I', 'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6']
    ]
    
    lead_indices = [
        [0, 3, 6, 9],
        [1, 4, 7, 10],
        [2, 5, 8, 11]
    ]
    
    # Apply preprocessing if requested
    if apply_preprocessing:
        ecg_processed = ecg[:, 150:-50]
        ecg_processed = signal.resample(ecg_processed, 200, axis=1)
        # Normalize each lead (same as training)
        max_val = np.max(np.abs(ecg_processed), axis=1, keepdims=True)
        max_val = np.where(max_val == 0, 1, max_val)  # Replace zeros with 1
        ecg_processed = ecg_processed / max_val
        sample_axis = np.arange(ecg_processed.shape[1])
        ecg_to_plot = ecg_processed
        title_suffix = " (Preprocessed)"
    else:
        sample_axis = np.arange(ecg.shape[1])
        ecg_to_plot = ecg
        title_suffix = " (Original)"
    
    # Create figure with 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle(f'12-Lead ECG Signal: {filename}{title_suffix}', fontsize=16, fontweight='bold')
    
    # Plot each lead in grid
    for row in range(3):
        for col in range(4):
            ax = axes[row, col]
            lead_idx = lead_indices[row][col]
            lead_name = lead_names[row][col]
            
            # Check if lead has all zeros
            if np.all(ecg_to_plot[lead_idx] == 0):
                ax.plot(sample_axis, ecg_to_plot[lead_idx], 'r-', linewidth=1.5, alpha=0.7)
                ax.set_title(f'{lead_name}', fontsize=12, fontweight='bold', color='red')
            else:
                ax.plot(sample_axis, ecg_to_plot[lead_idx], 'b-', linewidth=1.5)
                ax.set_title(lead_name, fontsize=12, fontweight='bold')
                # Show statistics
                max_val = np.max(np.abs(ecg_to_plot[lead_idx]))
                ax.text(0.98, 0.95, f'{max_val:.2f}', 
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            
            if row == 2:
                ax.set_xlabel('Samples', fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel('Amplitude (mV)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig, axes


if __name__ == '__main__':
    # Example usage
    
    # Set the path to your preprocessed data directory
    data_dir = 'cad_dataset_preprocessed/'
    
    # List available files
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    if len(files) == 0:
        print(f"No .npy files found in {data_dir}")
    else:
        # Show first few available files
        print(f"Found {len(files)} ECG files.")
        print("\nFirst 10 available files:")
        for i, f in enumerate(files[:10]):
            print(f"  {i+1}. {f}")
        
        # Select a file to visualize (you can change this)
        selected_file = files[0]
        file_path = os.path.join(data_dir, selected_file)
        
        print(f"\nVisualizing: {selected_file}")
        
        # Create grid visualizations only
        print("\nCreating grid view (original)...")
        fig1, _ = visualize_ecg_grid(file_path, apply_preprocessing=False)
        
        print("Creating grid view (preprocessed)...")
        fig2, _ = visualize_ecg_grid(file_path, apply_preprocessing=True)
        
        print("\nDisplaying figures...")
        print("  - Figure 1: 3x4 grid view (original)")
        print("  - Figure 2: 3x4 grid view (preprocessed)")
        
        # Display the plots
        plt.show()

