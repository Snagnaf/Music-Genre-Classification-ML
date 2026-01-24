from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from variables import print_step, print_success
import joblib
import os

# RANDOM FOREST MODEL TRAINING ==================================================
def train_rf(X_train, y_train, X_test, y_test):
    """
    Trains a Random Forest model (included in Pipeline for uniformity).

    Args:
        X_train: Training features (raw)
        y_train: Training labels (encoded)
        X_test: Test features (raw)
        y_test: Test labels (encoded)

    Returns:
        tuple: (y_pred_rf, rf_pipeline)
            - y_pred_rf: Predictions on the test set
            - rf_pipeline: Trained pipeline
    """
    print_step("[MODULE 2] Training RF...")

    # Pipeline Creation
    # Although RF does not strictly require scaling, we do it for uniformity
    rf_pipeline = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,   # Number of decision trees
            random_state=42,    # Reproducibility
            n_jobs=-1           # Use all available cores
        )
    )

    # Training
    rf_pipeline.fit(X_train, y_train)

    # Prediction
    y_pred_rf = rf_pipeline.predict(X_test)
    y_proba_rf = rf_pipeline.predict_proba(X_test)
    print_success(f"✓ Accuracy: {rf_pipeline.score(X_test, y_test):.4f}")

    # Save the model to the cache folder
    os.makedirs("src/cache", exist_ok=True)
    joblib.dump(rf_pipeline, "src/cache/rf_model.pkl")
    print_success("✓ Random Forest model saved in cache.")

    return y_pred_rf, rf_pipeline, y_proba_rf