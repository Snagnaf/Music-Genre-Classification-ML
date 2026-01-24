import os
import numpy as np
from variables import FEATURE_FILE, print_step, print_success, print_warning
from application.features import extract_features
from application.training import train_models
from application.dashboard import generate_dashboard

# MAIN FUNCTION ==================================================
def main():
    """
    Workflow:
    1. Loads features from the cache file if it exists, otherwise extracts them.
    2. Prepares data and trains models (SVM, Random Forest, CNN).
    3. Generates a complete dashboard with visualizations and metrics.
    """
    
    # PHASE 1: FEATURE LOADING OR EXTRACTION
    # Create cache directory if it doesn't exist
    os.makedirs("src/cache", exist_ok=True)
    
    X, X_spec, y = None, None, None

    if os.path.exists(FEATURE_FILE):
        try:
            # Attempt to load previously extracted features (cache)
            print_step("Loading data saved in cache...")
            data = np.load(FEATURE_FILE, allow_pickle=True)
            X, X_spec, y = data["X"], data["X_spec"], data["y"]
            print_success(f"✓ Loaded {len(X)} features from cache.")
        except Exception as e:
            print_warning(f"! Cache file is corrupted or incompatible: {e}")
            print_step("Re-extracting features from source")
            X, X_spec, y = None, None, None

    if X is None:
        # Extract features from dataset
        X, X_spec, y = extract_features()
        print_step("Saving features to cache...")
        np.savez_compressed(FEATURE_FILE, X=X, X_spec=X_spec, y=y)
        print_success("✓ Features saved to cache.")

    # PHASE 2: MODEL TRAINING
    print_step("Training models...")
    results = train_models(X, X_spec, y)

    # PHASE 3: DASHBOARD GENERATION
    print_step("Generating dashboard...")
    generate_dashboard(results)
   
# ENTRY POINT ==================================================
if __name__ == "__main__":
    main()
