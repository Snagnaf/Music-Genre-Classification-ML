from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from application.models.svm_model import train_svm
from application.models.rf_model import train_rf
from application.models.cnn_model import train_cnn
from variables import print_success
import joblib
import os

# DATA PREPARATION AND TRAINING ==================================================
def train_models(X, X_spec, y):
    """
    Complete process:
    1. Encoding: Converts textual labels into numbers.
    2. Split: Divides data into training (80%) and test (20%).
    3. Scaling: Standardizes features (mean=0, std=1).
    4. Training: Trains SVM and Random Forest.
    5. Evaluation: Calculates predictions and accuracy.
    
    Args:
        X: Feature array with shape (n_samples, n_features)
        y: Label array with shape (n_samples)
    
    Returns:
        Dictionary containing:
            - "X_train": Training features (raw)
            - "y_train": Training labels (encoded)
            - "y_test": Test labels (encoded)
            - "y_pred_svm": SVM model predictions
            - "y_pred_rf": Random Forest model predictions
            - "y_pred_cnn": CNN model predictions
            - "y_proba_svm": SVM probabilities
            - "y_proba_rf": RF probabilities
            - "y_proba_cnn": CNN probabilities
            - "history": CNN training history (for charts)
            - "label": LabelEncoder used (to decode labels)
            - "rf": Trained Random Forest model (for feature importance)
    """
    
    # PHASE 1: LABEL ENCODING
    # Transforms genre names into numbers (e.g., "Pop" -> 0, "Rock" -> 1, "Jazz" -> 2, ...)
    label = LabelEncoder()
    y_enc = label.fit_transform(y)
    print(f"{len(label.classes_)} genres found: {', '.join(label.classes_)}")

    # PHASE 2: TRAINING/TEST SPLIT
    # We split both standard features and spectrograms at the same time
    # Distribution: 80% Training, 10% Validation, 10% Test
    X_train, X_temp, X_spec_train, X_spec_temp, y_train, y_temp = train_test_split(
        X, X_spec, y_enc, 
        test_size=0.2, 
        stratify=y_enc, 
        random_state=42
    )
    
    # Split the temporary set (20%) into Test (10%) and Validation (10%)
    X_test, X_val, X_spec_test, X_spec_val, y_test, y_val = train_test_split(
        X_temp, X_spec_temp, y_temp, 
        test_size=0.5, 
        stratify=y_temp, 
        random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # PHASE 3: MODEL TRAINING
    # Note: Scaling is now handled internally by models (Pipeline / Normalization Layer)

    # SVM (Support Vector Machine)
    y_pred_svm, svm, y_proba_svm = train_svm(X_train, y_train, X_test, y_test)

    # Random Forest
    y_pred_rf, rf, y_proba_rf = train_rf(X_train, y_train, X_test, y_test)

    # CNN (Convolutional Neural Network)
    # Passing raw spectrograms, the internal normalization layer will handle it
    y_pred_cnn, cnn, history_cnn, y_proba_cnn = train_cnn(X_spec_train, y_train, X_spec_val, y_val, X_spec_test, y_test)

    # PHASE 5: RESULT PREPARATION
    # Save label encoder for future use
    os.makedirs("src/cache", exist_ok=True)
    joblib.dump(label, "src/cache/label_encoder.pkl")
    print_success("✓ LabelEncoder saved in cache.")

    return {
        "X_train": X_train,                     # Training features (raw)
        "y_train": y_train,                     # Training labels (encoded)
        "y_test": y_test,                       # Test labels (encoded)
        "y_pred_svm": y_pred_svm,               # SVM predictions
        "y_pred_rf": y_pred_rf,                 # Random Forest predictions
        "y_pred_cnn": y_pred_cnn,               # CNN predictions
        "y_proba_svm": y_proba_svm,             # SVM probabilities
        "y_proba_rf": y_proba_rf,               # RF probabilities
        "y_proba_cnn": y_proba_cnn,             # CNN probabilities
        "history": history_cnn,                 # CNN training history (for charts)
        "label": label,                         # LabelEncoder (to decode labels)
        "svm": svm,                             # Trained SVM model
        "rf": rf,                               # RF model (for feature importance)
    }
