from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from variables import print_step, print_success
import joblib
import os

# SVM MODEL TRAINING ==================================================
def train_svm(X_train, y_train, X_test, y_test):
    """
    Trains an SVM model with RBF kernel included in a Pipeline with StandardScaler.

    Args:
        X_train: Training features (raw)
        y_train: Training labels (encoded)
        X_test: Test features (raw)
        y_test: Test labels (encoded)

    Returns:
        tuple: (y_pred_svm, svm_pipeline)
            - y_pred_svm: Predictions on the test set
            - svm_pipeline: Trained pipeline (Scaler + SVM)
    """
    print_step("[MODULE 1] Training SVM...")

    # Pipeline Creation: Scaler -> SVM
    svm_pipeline = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",      # Radial Basis Function (default)
            C=10,              # Regularization parameter
            probability=True,  # Enable probability estimates (needed for ensemble voting)
            random_state=42    # Reproducibility
        )
    )

    # Training (scaler fits automatically here)
    svm_pipeline.fit(X_train, y_train)

    # Prediction (data is scaled automatically)
    y_pred_svm = svm_pipeline.predict(X_test)
    y_proba_svm = svm_pipeline.predict_proba(X_test)
    print_success(f"✓ Accuracy: {svm_pipeline.score(X_test, y_test):.4f}")

    # Save the model to the cache folder
    os.makedirs("src/cache", exist_ok=True)
    joblib.dump(svm_pipeline, "src/cache/svm_model.pkl")
    print_success("✓ SVM model saved in cache.")

    return y_pred_svm, svm_pipeline, y_proba_svm