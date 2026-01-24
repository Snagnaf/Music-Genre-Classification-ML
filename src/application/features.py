import os
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from variables import SAMPLE_RATE, DURATION, NUM_SEGMENTS, DATASET_PATH, print_step, print_success, print_warning, print_error

# FEATURE EXTRACTION ==================================================
def _process_file_worker(args):
    """
    Worker function to process a single audio file and extract features.
    
    Args:
        args (tuple): A tuple containing (file_path, genre, samples_per_segment, is_single_file_mode).
        
    Returns:
        tuple: (local_X, local_X_spec, local_y) containing extracted features lists.
    """
    file_path, genre, samples_per_segment, is_single_file_mode = args
    
    local_X, local_X_spec, local_y = [], [], []
    
    try:
        # Load audio (res_type='kaiser_fast' is faster)
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, res_type="kaiser_fast")
        
        # Determine number of segments
        num_segments = NUM_SEGMENTS if not is_single_file_mode else 1
        
        for s in range(num_segments):
            try:
                if not is_single_file_mode:
                    # Dataset: segments
                    start = samples_per_segment * s
                    finish = samples_per_segment * (s + 1)
                    segment = audio[start:finish]
                    
                    # Pad if too short
                    if len(segment) < samples_per_segment:
                        segment = librosa.util.fix_length(segment, size=samples_per_segment)
                else:
                    # Single file: use all
                    segment = audio

                # 1. TIME DOMAIN FEATURES
                # Zero Crossing Rate (indicates 'noisiness' or percussiveness)
                zcr = librosa.feature.zero_crossing_rate(y=segment)
                
                # Root Mean Square: Mean volume/energy of the segment
                rms = librosa.feature.rms(y=segment)
                
                # Estimate BPM (beats per minute) of the track
                tempo, _ = librosa.beat.beat_track(y=segment, sr=SAMPLE_RATE)
                
                # Amplitude Variance: Indicates dynamic range
                amp_var = np.var(segment)

                # 2. FREQUENCY DOMAIN FEATURES
                # Mel-Frequency Cepstral Coefficients: "Timbral" fingerprint (e.g. guitar vs piano)
                mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=13)
                
                # Chroma: Energy distribution over the 12 musical notes (C, C#, etc.) - useful for harmony
                chroma = librosa.feature.chroma_stft(y=segment, sr=SAMPLE_RATE)
                
                # Spectral Centroid: Indicates how "bright" or "dark" the sound is (weighted mean frequency)
                spec_cent = librosa.feature.spectral_centroid(y=segment, sr=SAMPLE_RATE)
                
                # Spectral Bandwidth: Indicates the width of the frequency range
                spec_bw = librosa.feature.spectral_bandwidth(y=segment, sr=SAMPLE_RATE)
                
                # Spectral Contrast: Difference between peaks and valleys in the spectrum (distinguishes pure sounds from noise)
                spec_con = librosa.feature.spectral_contrast(y=segment, sr=SAMPLE_RATE)
                
                # Spectral Rolloff: Frequency below which 85% of the energy lies (distinguishes harmonic from noisy sounds)
                spec_roll = librosa.feature.spectral_rolloff(y=segment, sr=SAMPLE_RATE)

                # 3. SPECTROGRAM
                # Mel Spectrogram: Visual representation of energy on a logarithmic scale (Mel) similar to human hearing
                melspec = librosa.feature.melspectrogram(y=segment, sr=SAMPLE_RATE, n_mels=128)
                melspec_db = librosa.power_to_db(melspec, ref=np.max)
                
                if melspec_db.shape != (128, 130):
                    melspec_db = librosa.util.fix_length(melspec_db, size=130, axis=1)

                # Feature Vector
                feat_vector = np.hstack([
                    np.mean(zcr), np.mean(rms), tempo, amp_var,
                    np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                    np.mean(chroma, axis=1), np.mean(spec_cent),
                    np.mean(spec_bw), np.mean(spec_con, axis=1),
                    np.mean(spec_roll)
                ])

                local_X.append(feat_vector)
                local_X_spec.append(melspec_db)
                local_y.append(genre)
            except Exception:
                continue

    except Exception as e:
        print_error(f"Error processing \"{file_path}\": {e}")
        pass

    return local_X, local_X_spec, local_y


# FEATURE EXTRACTION ==================================================
def extract_features(file_path=None, return_mean=False):
    """ 
    This function:
    1. If file_path is None: Recursively scans the dataset folder.
    2. If file_path is provided: Extracts features only from that file.
    3. For each file, loads the audio and divides it into segments.
    4. For each segment, extracts several audio features.
    5. Returns features and labels (genres).
    
    Args:
        file_path (str, optional): Path to a single audio file. If None, uses the entire dataset.
        return_mean (bool): If True and file_path is provided, returns the mean of features instead of all segments.
    
    Returns:
        - X: Numpy array of shape (n_samples, n_features) containing features
        - X_spec: Numpy array of spectrograms for CNN
        - y: Numpy array of shape (n_samples,) containing labels (genres)
    
    Notes:
        Each track is divided into NUM_SEGMENTS segments to increase data.
        Each feature vector contains:
        - 13 values: Mean of MFCC coefficients
        - 13 values: Standard deviation of MFCC coefficients
        - 12 values: Chroma coefficients
        - 1 value: Spectral Centroid
        - 1 value: Tempo/BPM
        Total: 40 features per segment
    """
    
    samples_per_segment = int((SAMPLE_RATE * DURATION) / NUM_SEGMENTS)
    X, X_spec, y = [], [], []

    # PHASE 1: FILE GATHERING
    if file_path is not None:
        if os.path.exists(file_path):
            all_files = [(file_path, "unknown")]
        else:
            raise ValueError(f"File {file_path} not found")
    else:
        print_step("Scanning dataset...")
        all_files = []
        for root, _, filenames in os.walk(DATASET_PATH):
            genre = root.split(os.path.sep)[-1]
            if not genre or genre == DATASET_PATH.split(os.path.sep)[-1]:
                continue
            for f in filenames:
                if f.endswith((".wav", ".mp3")):
                    all_files.append((os.path.join(root, f), genre))

        print(f"Found {len(all_files)} files. Starting extraction...")

    # Preparing arguments
    is_single_mode = file_path is not None
    worker_args = [(f, g, samples_per_segment, is_single_mode) for f, g in all_files]

    # PHASE 2: EXECUTION (PARALLEL OR SEQUENTIAL)
    if is_single_mode:
        # Direct execution for single file (avoids process overhead)
        results = [_process_file_worker(worker_args[0])]
    else:
        # Parallel execution
        # Determine safe number of workers
        max_workers = max(1, (os.cpu_count() or 1) - 1) # Leave one core free for system
        
        print(f"Using {max_workers} processes in parallel...")
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            futures = [executor.submit(_process_file_worker, arg) for arg in worker_args]
            
            # Collect results as they finish with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), unit="file", colour="blue"):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"Error in a worker: {e}")

    # PHASE 3: AGGREGATION
    for res_X, res_X_spec, res_y in results:
        if res_X:
            X.extend(res_X)
            X_spec.extend(res_X_spec)
            y.extend(res_y)

    print_success(f"✓ Successfully extracted {len(X)} features.")
    if len(X) == 0:
        raise ValueError("No features extracted.")
    
    print(f"Number of features per sample: {len(X[0])}")

    if return_mean and file_path is not None:
        X = np.mean(X, axis=0, keepdims=True)
        X_spec = np.mean(X_spec, axis=0, keepdims=True)
        y = np.array(["unknown"])

    return np.array(X), np.array(X_spec), np.array(y)
