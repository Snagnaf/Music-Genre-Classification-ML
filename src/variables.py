import os
import warnings

# CONSOLE COLORS
class Col:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_step(msg):
    """Prints a step header message."""
    print(f"\n{Col.HEADER}{Col.BOLD}{msg}{Col.ENDC}")

def print_success(msg):
    """Prints a success message."""
    print(f"{Col.OKGREEN}{msg}{Col.ENDC}")

def print_testing(msg):
    """Prints a message related to testing songs."""
    print(f"{Col.OKCYAN}{Col.BOLD}{msg}{Col.ENDC}")

def print_warning(msg):
    """Prints a warning message."""
    print(f"{Col.WARNING}{msg}{Col.ENDC}")

def print_error(msg):
    """Prints an error message."""
    print(f"\n{Col.FAIL}{Col.BOLD}{msg}{Col.ENDC}")

# PARALLELISM OPTIMIZATION
# Disable NumPy/MKL/OpenMP internal multithreading to avoid
# every worker process trying to use all cores, saturating the CPU.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# SILENCE ANNOYING TENSORFLOW LOGS
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Silence specific librosa/audioread warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", module="librosa.core.audio")

# FILE PATHS AND DATASET
# Path to the folder containing musical genres organized in subfolders
DATASET_PATH = "src/data/genres_original"

# Filename where extracted features are saved (cache)
FEATURE_FILE = f"src/cache/music_features.npz"

# Filename for the classification dashboard
DASHBOARD_FILE = f"src/images/dashboard.png"

# AUDIO TECHNICAL PARAMETERS
# Standard sampling frequency (Hz)
# Common value for music audio: 22050 Hz
SAMPLE_RATE = 22050
# Duration of each music track in seconds
# The GTZAN dataset contains 30-second tracks
DURATION = 30
# Number of segments to divide each track into
# Each track is divided into 10 segments of 3 seconds each
# This increases the data available for training (Data Augmentation)
NUM_SEGMENTS = 10

# EXTRACTED FEATURE NAMES
# Total features: 52
# Breakdown: 13 MFCC (mean) + 13 MFCC (std) + 12 Chroma + 7 Spec Contrast + 7 others = 52
FEATURE_NAMES = (
    # TIME DOMAIN FEATURES
    ["Zero_Crossing_Rate",          # Zero Crossing Rate    
    "RMS_Energy",                   # RMS Energy  
    "tempo",                        # Tempo
    "Central Moments"] +            # Central Moments (Variance of amplitude)

    # FREQUENCY DOMAIN FEATURES
    # MFCC Coefficients (Mel-Frequency Cepstral Coefficients), capture sound "timbre" - 13 coeffs for mean and std dev
    [f"MFCC_Mean_{i}" for i in range(13)] +      # Mean of MFCC coefficients
    [f"MFCC_Std_{i}" for i in range(13)] +       # Standard Deviation of MFCC coefficients
    
    # Chroma Coefficients
    # Capture harmony and musical notes (crucial for Jazz/Classical)
    [f"Chroma_{i}" for i in range(12)] +          # 12 chromatic notes
    
    # Other features
    ["Spec_Centroid",           # Spectral Centroid (sound brightness)
    "Spectral Bandwidth",       # Spectral Bandwidth
    ] + [f"Spectral_Contrast_{i}" for i in range(7)] + [ # 7 frequency bands
    "Spectral Rolloff"]         # Spectral Rolloff
)
