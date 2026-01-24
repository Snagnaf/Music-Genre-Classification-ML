from variables import print_step, print_success, print_warning, print_error, print_testing
import os
import numpy as np
import joblib
from tensorflow import keras
from application.features import extract_features

# TEST MODEL ON NEW SONGS ==================================================
def test_models(song_path):
    """
    Loads models from cache and predicts the song's genre.

    Args:
        song_path (str): Path to the song to test
    """
    # Extract ALL segments from the song (don't average them yet)
    X_segments, X_spec_segments, _ = extract_features(file_path=song_path, return_mean=False)

    # Load labels and models
    label_encoder = joblib.load("src/cache/label_encoder.pkl")
    svm_model = joblib.load("src/cache/svm_model.pkl")
    rf_model = joblib.load("src/cache/rf_model.pkl")
    cnn_model = keras.models.load_model("src/cache/cnn_model.keras")

    # --- ENSEMBLE PREDICTION (VOTING) ---
    # We predict each segment and take the average of probabilities
    # 1. SVM & RF (using average of results or probabilities)
    # Most sklearn models in pipelines support predict_proba
    probs_svm = svm_model.predict_proba(X_segments)
    probs_rf = rf_model.predict_proba(X_segments)
    
    # 2. CNN (predict probabilities for each spectrogram segment)
    spec_input = X_spec_segments[..., np.newaxis]
    probs_cnn = cnn_model.predict(spec_input, verbose=0)

    # Final Average Probabilities
    mean_probs_svm = np.mean(probs_svm, axis=0)
    mean_probs_rf = np.mean(probs_rf, axis=0)
    mean_probs_cnn = np.mean(probs_cnn, axis=0)

    # Final Genre (Index with highest probability)
    genre_svm = label_encoder.inverse_transform([np.argmax(mean_probs_svm)])[0]
    genre_rf = label_encoder.inverse_transform([np.argmax(mean_probs_rf)])[0]
    genre_cnn = label_encoder.inverse_transform([np.argmax(mean_probs_cnn)])[0]

    # Print results
    print_step("PREDICTION RESULTS")
    print_success(f"SVM: {genre_svm} ({np.max(mean_probs_svm)*100:.1f}%)")
    print_success(f"Random Forest: {genre_rf} ({np.max(mean_probs_rf)*100:.1f}%)")
    print_success(f"CNN: {genre_cnn} ({np.max(mean_probs_cnn)*100:.1f}%)")

# ENTRY POINT ==================================================
if __name__ == "__main__":
    # Find all songs in the song folder
    song_dir = "src/songs"
    
    # Check models
    required_files = [
        "src/cache/label_encoder.pkl",
        "src/cache/svm_model.pkl",
        "src/cache/rf_model.pkl",
        "src/cache/cnn_model.keras"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print_error("\nModels not found, run full training first with: python src/main.py")
        exit()

    # Check songs
    if not os.path.exists(song_dir):
        print_error(f"Folder {song_dir} does not exist. Create the folder and add songs.")
        exit()

    songs = [f for f in os.listdir(song_dir) if f.endswith(('.wav', '.mp3'))]
    if not songs:
        print_warning(f"No songs found in {song_dir}. Add some audio files.")
        exit()

    print_testing(f"\nFound {len(songs)} songs to test:")
    for song_file in songs:
        print(f"- {song_file}")

    # Test each song
    for i, song_file in enumerate(songs, 1):
        song_path = os.path.join(song_dir, song_file)
        print_testing(f"\n[{i}/{len(songs)}] Testing: {song_file}")
        try:
            test_models(song_path)
        except Exception as e:
            print_error(f"Error testing {song_file}: {e}")
