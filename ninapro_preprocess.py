"""
ninapro_batch_preprocess.py
Batch processes all 10 subjects from Ninapro DB5 (Exercise 3: Grasping).
Outputs a single, massive ML-ready CSV file for PyTorch.
"""
import scipy.io as sio
import scipy.signal as signal
import numpy as np
import pandas as pd
import os
import glob

def process_single_subject(mat_file_path, subject_id):
    print(f"Processing Subject {subject_id}: {mat_file_path}...")
    
    try:
        data = sio.loadmat(mat_file_path)
    except Exception as e:
        print(f"  -> Error loading {mat_file_path}. Skipping.")
        return None

    emg_raw = data['emg']               # 16 channels
    glove_raw = data['glove']           # 22 channels
    labels = data['restimulus']         # Refined labels
    
    sfreq = 200 
    window_size = int(0.200 * sfreq)    # 40 samples (200ms)
    stride = int(0.050 * sfreq)         # 10 samples (50ms)

    # 1. SIGNAL PROCESSING (Butterworth Filter on EMG)
    emg_rectified = np.abs(emg_raw)
    nyquist = 0.5 * sfreq
    cutoff = 5.0 / nyquist 
    b, a = signal.butter(3, cutoff, btype='low')
    
    emg_filtered = np.zeros_like(emg_rectified)
    for i in range(emg_filtered.shape[1]): 
        emg_filtered[:, i] = signal.filtfilt(b, a, emg_rectified[:, i])

    # 2. OVERLAPPING SLIDING WINDOWS
    processed_windows = []
    num_samples = emg_filtered.shape[0]
    
    for start in range(0, num_samples - window_size, stride):
        end = start + window_size
        
        # Features: Mean Absolute Value
        emg_features = np.mean(emg_filtered[start:end, :], axis=0)
        glove_features = np.mean(glove_raw[start:end, :], axis=0)
        
        # Predominant Label
        window_labels = labels[start:end].flatten()
        predominant_label = np.bincount(window_labels).argmax()
        
        # Row: [Subject_ID] + [16 EMG] + [22 Glove] + [Label]
        row = [subject_id] + list(emg_features) + list(glove_features) + [predominant_label]
        processed_windows.append(row)

    return processed_windows

# ==========================================
# MASTER EXECUTION
# ==========================================
if __name__ == "__main__":
    raw_data_dir = "DB5_Raw_Data" # Make sure your .mat files are in this folder!
    output_csv = "DB5_Master_Grasp_Dataset.csv"
    
    all_processed_data = []
    
    # Find all Exercise 3 .mat files in the directory
    search_pattern = os.path.join(raw_data_dir, "*E3*.mat")
    mat_files = glob.glob(search_pattern)
    
    if not mat_files:
        print(f"No Exercise 3 .mat files found in '{raw_data_dir}'. Please organize the files.")
        exit()
        
    print(f"Found {len(mat_files)} subjects. Starting Batch Processing...")

    for file_path in mat_files:
        # Extract subject ID from filename (Assuming format like S1_E3_A1.mat)
        filename = os.path.basename(file_path)
        try:
            # Parses '1' out of 'S1_E3_A1.mat'
            subject_id = int(filename.split('_')[0].replace('S', '').replace('s', ''))
        except:
            subject_id = 99 # Fallback if naming is weird
            
        subject_data = process_single_subject(file_path, subject_id)
        
        if subject_data is not None:
            all_processed_data.extend(subject_data)
            print(f"  -> Generated {len(subject_data)} ML windows.")

    # Convert everything to a single DataFrame
    emg_cols = [f'emg_ch{i+1}' for i in range(16)]
    glove_cols = [f'glove_j{i+1}' for i in range(22)]
    cols = ['subject_id'] + emg_cols + glove_cols + ['gesture_class']
    
    df = pd.DataFrame(all_processed_data, columns=cols)
    
    print("\nSaving Master Dataset...")
    df.to_csv(output_csv, index=False)
    print(f"\n========================================")
    print(f"SUCCESS! Master Dataset created: {output_csv}")
    print(f"Total Rows (Sliding Windows): {len(df):,}")
    print(f"Total Columns (Features): {len(df.columns)}")
    print(f"========================================")