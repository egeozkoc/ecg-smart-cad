import xmltodict
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import glob
import os
import scipy.io
from scipy import signal

def extract_representative_beats(xml_filename: str, downsample_to_250hz: bool = False):
    """
    Extract representative beats from Philips SierraECG XML file.
    
    Args:
        xml_filename: Path to the Philips SierraECG XML file
        downsample_to_250hz: If True, downsample from 1000 Hz to 250 Hz (800 samples -> 200 samples).
                             If False, keep the original 800 samples (1000 Hz).
        
    Returns:
        Dictionary containing:
        - 'beats': numpy array of shape (12, 800) with representative beats for each lead
        - 'metadata': Sampling rate, signal resolution, etc.
        - 'lead_names': List of lead names
        - 'format': Detected format type
    """
    import numpy as np
    import xmltodict
    from scipy import signal
    import os

    # Parse XML file
    with open(xml_filename, 'rb') as f:
        dic = xmltodict.parse(f)
    
    # Check if it's a Philips SierraECG file
    if 'restingecgdata' not in dic:
        raise ValueError("Not a Philips SierraECG XML file")
    
    # Extract metadata
    metadata = {}
    
    # Extract repbeats section
    if 'repbeats' not in dic['restingecgdata']['waveforms']:
        raise ValueError("No representative beats found in Philips XML file")
    
    repbeats = dic['restingecgdata']['waveforms']['repbeats']['repbeat']
    
    # Get sampling rate from repbeats section (can be different from main signal)
    if '@samplespersec' in dic['restingecgdata']['waveforms']['repbeats']:
        metadata['sampling_rate'] = int(dic['restingecgdata']['waveforms']['repbeats']['@samplespersec'])
    else:
        # Fallback to main signal sampling rate
        metadata['sampling_rate'] = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['samplingrate'])
    
    # Get resolution from repbeats section if available
    if '@resolution' in dic['restingecgdata']['waveforms']['repbeats']:
        metadata['signal_resolution'] = float(dic['restingecgdata']['waveforms']['repbeats']['@resolution'])
    else:
        # Fallback to main signal resolution
        if 'signalresolution' in dic['restingecgdata']['dataacquisition']['signalcharacteristics']:
            metadata['signal_resolution'] = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['signalresolution'])
        else:
            metadata['signal_resolution'] = int(dic['restingecgdata']['dataacquisition']['signalcharacteristics']['resolution'])
    
    # Get actual duration from the first repbeat's data
    first_repbeat = repbeats[0]
    if 'waveform' in first_repbeat:
        raw_data = first_repbeat['waveform']['#text'].strip().split()
    else:
        # Fallback to direct #text if waveform tag doesn't exist
        raw_data = first_repbeat['#text'].strip().split()
    
    # Calculate actual duration from the data
    duration = len(raw_data)
    metadata['beat_duration'] = 800  # Fixed to 800 samples
    
    # Initialize arrays
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_leads = 12
    num_samples = 200 if downsample_to_250hz else 800
    # First, build a (12, N) array of all leads' raw data
    all_leads_data = []
    for lead_name in lead_names:
        # Find the repbeat for this lead
        repbeat = next(rb for rb in repbeats if rb['@leadname'] == lead_name)
        if 'waveform' in repbeat:
            raw_data = repbeat['waveform']['#text'].strip().split()
        else:
            raw_data = repbeat['#text'].strip().split()
        beat_data = np.array([int(x) for x in raw_data]) * metadata['signal_resolution']
        # Check for zero-padded beats (2.4s with last 1.2s zeros)
        if len(beat_data) == 2400:
            half = 1200
            if np.all(beat_data[half:] == 0):
                beat_data = beat_data[:half]
        all_leads_data.append(beat_data)
    all_leads_data = np.stack(all_leads_data, axis=0)  # shape (12, N)
    N = all_leads_data.shape[1]
    # Compute RMS across all leads
    rms = np.sqrt(np.mean(all_leads_data**2, axis=0))
    search_end = N // 2
    rpeak = np.argmax(np.abs(rms[:search_end]))
    
    if debug_mode:
        fig = plt.figure(figsize=(12,4))
        plt.plot(rms)
        plt.axvline(x=rpeak, color='r', linestyle='--', label='R peak')
        plt.title(f'RMS Signal with Detected R Peak\n{os.path.basename(xml_filename)}')
        plt.xlabel('Sample')
        plt.ylabel('RMS Amplitude')
        plt.legend()
        plt.show()
        plt.close(fig)

        
    # Extract window for all leads
    start = max(0, rpeak - 300)
    end = min(N, rpeak + 500)
    window_rpeak_index = rpeak - start  # Index of R peak in the windowed beat
    beats = np.zeros((num_leads, num_samples))
    for lead in range(num_leads):
        beat = all_leads_data[lead, start:end]
        # Pad if needed
        if len(beat) < 800:
            if start == 0:
                beat = np.pad(beat, (800 - len(beat), 0))
            else:
                beat = np.pad(beat, (0, 800 - len(beat)))
        # Anti-aliasing filter and downsample
        if downsample_to_250hz:
            # Design and apply Butterworth lowpass filter
            nyquist = 500.0  # Nyquist frequency (1000 Hz / 2)
            cutoff = 100.0  # Cutoff frequency in Hz
            normalized_cutoff = cutoff / nyquist
            b, a = signal.butter(5, normalized_cutoff, btype='low')
            beat = signal.filtfilt(b, a, beat)
            
            # Downsample from 1000 Hz to 250 Hz
            beat = signal.resample(beat, 200)
        beats[lead, :] = beat
    # Normalize each lead
    max_val = np.max(np.abs(beats), axis=1)
    max_val[max_val == 0] = 1  # avoid division by zero
    beats = beats / max_val[:, None]
    # Update metadata
    metadata['sampling_rate'] = 250 if downsample_to_250hz else 1000
    metadata['beat_duration'] = num_samples
    return {
        'beats': beats,
        'metadata': metadata,
        'lead_names': lead_names,
        'format': 'Philips SierraECG',
        'global_rpeak_index': window_rpeak_index  # Use the index within the windowed beat
    }

def plot_representative_beats(beats_data, xml_filename=None):
    """
    Plot the representative beats for all 12 leads.
    
    Args:
        beats_data: Dictionary returned by extract_representative_beats()
        xml_filename: Optional filename to include in plot title
    """
    
    beats = beats_data['beats']
    lead_names = beats_data['lead_names']
    metadata = beats_data['metadata']
    format_type = beats_data.get('format', 'Unknown')
    # global_rpeak_index = beats_data.get('global_rpeak_index', None)  # No longer needed
    
    # Create time axis
    duration_sec = metadata['beat_duration'] / metadata['sampling_rate']
    time_axis = np.linspace(0, duration_sec, metadata['beat_duration'])
    
    # Create subplot grid: 4 rows x 3 columns for 12 leads
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    
    # Create title based on whether filename is provided
    if xml_filename:
        title = f'Representative ECG Beats - {format_type}\n{os.path.basename(xml_filename)} ({duration_sec:.3f} seconds)'
    else:
        title = f'Representative ECG Beats - {format_type} ({duration_sec:.3f} seconds)'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot each lead
    for i, (lead_name, lead_data) in enumerate(zip(lead_names, beats)):
        ax = axes[i//3, i%3]
        
        # Plot the signal
        ax.plot(time_axis, lead_data, 'b-', linewidth=1.5)
        
        # Customize the plot
        ax.set_title(f'Lead {lead_name}', fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (μV)')
        ax.grid(True, alpha=0.3)
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Set reasonable y-axis limits
        y_range = np.max(lead_data) - np.min(lead_data)
        y_margin = y_range * 0.1
        ax.set_ylim(np.min(lead_data) - y_margin, np.max(lead_data) + y_margin)
        # No legend needed
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Allow hardcoded path or use folder selection dialog
    folder_path = 'ecg_data/registry'  # Set your hardcoded path here, e.g. "C:/ECG_Data"
    debug_mode = False  # Set to True to enable debug printing and plotting
    
    if not folder_path:
        # Ask user to select folder containing ECG data
        root = tk.Tk()
        root.withdraw()
        
        # Open folder selection dialog
        folder_path = filedialog.askdirectory(
            title="Select folder containing ECG data"
        )
        
        if not folder_path:
            print("No folder selected")
            exit()
    
    print(f"Using folder: {folder_path}")

    # Get list of XML files in folder     
    xml_files = glob.glob(os.path.join(folder_path, "*.xml"))
    
    if not xml_files:
        print("No XML files found in selected folder")
        exit()
    
    # Create output directory if it doesn't exist
    output_dir = "ecg_data/ecg_median_beats_from_xml"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize counters
    processed_count = 0
    skipped_count = 0
    
    # Process each XML file
    for xml_file in xml_files:
        if debug_mode:
            print(f"\nProcessing {os.path.basename(xml_file)}")
        try:
            # Try to extract representative beats
            result = extract_representative_beats(xml_file, downsample_to_250hz=True) # Default to downsampling
            
            # Save beats as .npy file
            output_filename = os.path.splitext(os.path.basename(xml_file))[0] + '.npy'
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, result['beats'])
            
            # Plot first 10 ECGs if in debug mode
            if debug_mode and processed_count < 10:
                sampling_rate = result['metadata']['sampling_rate']
                num_samples = result['metadata']['beat_duration']
                time_axis = np.linspace(0, num_samples / sampling_rate, num_samples, endpoint=False)
                # global_rpeak_index = result.get('global_rpeak_index', None)  # No longer needed

                fig, axes = plt.subplots(4, 3, figsize=(15, 12))
                fig.suptitle(f'ECG Data - {os.path.basename(xml_file)}', fontsize=16)
                
                # Plot each lead
                for i, (lead_name, lead_data) in enumerate(zip(result['lead_names'], result['beats'])):
                    ax = axes[i//3, i%3]
                    ax.plot(time_axis, lead_data)
                    ax.set_title(f'Lead {lead_name}')
                    ax.set_xlabel('Time (s)')
                    ax.grid(True)
                
                plt.tight_layout()
                plt.show()
                
            processed_count += 1
            if debug_mode:
                print(f"✓ Successfully processed and saved {os.path.basename(xml_file)}")
                # Preview the saved data
                loaded_data = np.load(output_path)
                print(f"Data shape: {loaded_data.shape}")
                for lead_idx, lead_name in enumerate(result['lead_names']):
                    print(f"Lead {lead_name} first 5 values: {loaded_data[lead_idx, :5]}")
            
        except ValueError as e:
            skipped_count += 1
            if debug_mode:
                print(f"✗ Skipping file - {str(e)}")
        except Exception as e:
            skipped_count += 1
            if debug_mode:
                print(f"✗ Error processing file: {str(e)}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total XML files found: {len(xml_files)}")
    print(f"Files successfully processed: {processed_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"{'='*50}")
    print(f"Median beats saved to: {output_dir}")
    print(f"{'='*50}")
