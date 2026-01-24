import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from variables import FEATURE_NAMES, DASHBOARD_FILE, print_step, print_success

# HELPER FUNCTION ==========================================================
def save_single_plot(filename, plot_func, *args, figsize=(10, 8), **kwargs):
    """
    Helper to save a single plot to a file string.
    """
    fig, ax = plt.subplots(figsize=figsize)
    plot_func(ax, *args, **kwargs)
    plt.tight_layout()
    # Ensure directory exists
    os.makedirs("src/images", exist_ok=True)
    path = os.path.join("src/images", filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)

# PCA ====================================================================
def plot_pca(ax, X_train, y_train, label_encoder):
    """
    Plots the PCA 2D distribution of the training data on the given axis.
    
    Args:
        ax: Matplotlib axes object
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        label_encoder (LabelEncoder): Encoder with class names
    """
    # PCA (Principal Component Analysis): Reduces dimensionality to 2D
    pca_scaler = StandardScaler()
    X_train_sc_pca = pca_scaler.fit_transform(X_train)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_sc_pca)
    
    # Plot each genre with a different color
    for i, target in enumerate(label_encoder.classes_):
        ax.scatter(
            X_pca[y_train == i, 0],  # First principal component
            X_pca[y_train == i, 1],  # Second principal component
            label=target,
            alpha=0.6,               # Transparency
            s=15                     # Point size
        )
    
    ax.set_title("PCA 2D Map: Genre Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.legend(fontsize="small", loc="upper right")

# FEATURE IMPORTANCE ========================================================
def plot_feature_importance(ax, rf_pipeline):
    """
    Plots the top 15 most important features from the Random Forest model.
    
    Args:
        ax: Matplotlib axes object
        rf_pipeline (Pipeline): Trained Random Forest pipeline
    """
    # Extracts feature importance from the Random Forest model
    importances = rf_pipeline.named_steps['randomforestclassifier'].feature_importances_

    # Select the top 15 most important features
    indices = np.argsort(importances)[-15:]

    # Create a horizontal bar chart
    ax.barh(range(len(indices)), importances[indices], color="teal")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in indices])
    ax.set_title("Top 15 Decisive Features (RF)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")

# CNN HISTORY ===============================================================
def plot_cnn_history(ax, history):
    """
    Plots the CNN training accuracy evolution.
    
    Args:
        ax: Matplotlib axes object
        history (History): Keras training history object
    """
    # Plot accuracy during training
    ax.plot(history.history["accuracy"], label="Training", linewidth=2)
    ax.set_title("CNN Learning Evolution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

# CONFUSION MATRIX ==========================================================
def plot_confusion_matrix(ax, y_true, y_pred, model_name, cmap, label_encoder, cbar=False):
    """
    Plots a confusion matrix for a specific model.
    
    Args:
        ax: Matplotlib axes object
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        model_name (str): Name of the model
        cmap (str): Colormap name for the heatmap
        label_encoder (LabelEncoder): Encoder with class names
        cbar (bool): Whether to show the colorbar. Defaults to False.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Visualize as heatmap
    sns.heatmap(
        cm,
        annot=True,                    # Show values in cells
        fmt="d",                       # Integer format
        xticklabels=label_encoder.classes_,      # X axis labels (genres)
        yticklabels=label_encoder.classes_,       # Y axis labels (genres)
        cmap=cmap,                     # Colormap
        ax=ax,
        cbar=cbar                      # Show/Hide colorbar
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# F1 SCORES =================================================================
def plot_f1_scores(ax, y_test, models_preds, label_encoder):
    """
    Plots a bar chart comparing F1-scores across models and genres.
    
    Args:
        ax: Matplotlib axes object
        y_test (np.ndarray): True test labels
        models_preds (list): List of tuples (model_name, prediction_array)
        label_encoder (LabelEncoder): Encoder with class names
    """
    # models_preds is a list of tuples: (model_name, prediction_array)
    metrics_data = []
    
    for m_name, preds in models_preds:
        # Get classification report as dictionary
        rep = classification_report(
            y_test, preds, target_names=label_encoder.classes_, output_dict=True
        )

        # Extract F1-Score for each genre
        for g in label_encoder.classes_:
            metrics_data.append({
                "Genre": g,
                "Model": m_name,
                "F1-Score": rep[g]["f1-score"]
            })

    # Create a DataFrame and visualize as barplot
    df_metrics = pd.DataFrame(metrics_data)
    sns.barplot(
        data=df_metrics,
        x="Genre",
        y="F1-Score",
        hue="Model",
        ax=ax,
        palette="viridis"
    )
    ax.set_title(
        "Overall Reliability (F1-Score) by Model and Genre",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Model", loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

# DASHBOARD GENERATION ==================================================
def generate_dashboard(results):
    """
    The dashboard is displayed and saved as a PNG file. Includes:
    - PCA 2D visualization of data
    - Feature Importance (Random Forest)
    - Confusion Matrices for all models
    - F1-Score comparison between models and genres
    - Detailed reports printed in the console
    
    Args:
        results (Dict[str, Any]): Dictionary containing model results and data.
    """
    
    # Set dark theme for the dashboard
    plt.style.use('dark_background')

    # Extract data from the results dictionary
    X_train = results["X_train"]
    y_train = results["y_train"]
    y_test = results["y_test"]
    y_pred_svm = results["y_pred_svm"]
    y_pred_rf = results["y_pred_rf"]
    y_pred_cnn = results["y_pred_cnn"]
    y_proba_svm = results["y_proba_svm"]
    y_proba_rf = results["y_proba_rf"]
    y_proba_cnn = results["y_proba_cnn"]
    history = results["history"]
    label = results["label"]
    rf_pipeline = results["rf"]

    # 1. SAVE INDIVIDUAL PLOTS
    # ========================
    print_step("Saving individual chart files...")

    # PCA
    save_single_plot("pca_map.png", plot_pca, X_train, y_train, label)
    
    # Feature Importance
    save_single_plot("feature_importance.png", plot_feature_importance, rf_pipeline)
    
    # CNN History
    save_single_plot("cnn_history.png", plot_cnn_history, history)

    # Confusion Matrices
    cm_configs = [
        (y_pred_svm, "SVM", "Blues", "cm_svm.png"),
        (y_pred_rf, "Random Forest", "Greens", "cm_rf.png"),
        (y_pred_cnn, "CNN", "Reds", "cm_cnn.png")
    ]
    for preds, name, cmap, fname in cm_configs:
        save_single_plot(fname, plot_confusion_matrix, y_test, preds, name, cmap, label, cbar=True)
    
    # F1 Scores
    models_preds = [
        ("SVM", y_pred_svm),
        ("Random Forest", y_pred_rf),
        ("CNN", y_pred_cnn),
    ]
    save_single_plot("f1_scores.png", plot_f1_scores, y_test, models_preds, label)

    print_success("✓ Individual charts saved in src/images/")


    # 2. CREATE AND SAVE MAIN DASHBOARD
    # =================================
    
    # Create a large figure to hold all plots
    fig = plt.figure(figsize=(22, 16))

    # Create a 3x3 grid to organize subplots
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

    # CHART 1: PCA 2D MAP
    ax1 = fig.add_subplot(gs[0, 0])
    plot_pca(ax1, X_train, y_train, label)

    # CHART 2: FEATURE IMPORTANCE (Random Forest)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_feature_importance(ax2, rf_pipeline)

    # CHART 3: CNN TRAINING HISTORY
    ax3 = fig.add_subplot(gs[0, 2])
    plot_cnn_history(ax3, history)

    # CHARTS 4-6: CONFUSION MATRICES
    plots = [
        (y_pred_svm, "SVM", "Blues", 0),
        (y_pred_rf, "Random Forest", "Greens", 1),
        (y_pred_cnn, "CNN", "Reds", 2)
    ]
    
    for pred, name, cmap, col in plots:
        ax = fig.add_subplot(gs[1, col])
        plot_confusion_matrix(ax, y_test, pred, name, cmap, label, cbar=False)

    # CHART 7: F1-SCORE COMPARISON
    ax7 = fig.add_subplot(gs[2, :])  # Occupies entire bottom row
    plot_f1_scores(ax7, y_test, models_preds, label)

    # FINALIZATION AND SAVING
    # Optimize layout to avoid overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create images directory if it doesn't exist
    os.makedirs("src/images", exist_ok=True)

    # Save figure as high-resolution PNG
    plt.savefig(DASHBOARD_FILE, dpi=150, bbox_inches="tight")
    print_success("✓ Main Dashboard saved in images.")
    
    # DETAILED REPORTS IN CONSOLE
    for m_name, preds, proba in [
        ("SVM", y_pred_svm, y_proba_svm),
        ("Random Forest", y_pred_rf, y_proba_rf),
        ("CNN", y_pred_cnn, y_proba_cnn)
    ]:

        print_step(f"\n{m_name}:")
        # Get classification report as dictionary
        rep = classification_report(y_test, preds, target_names=label.classes_, output_dict=True)
        accuracy = rep['accuracy']
        f1_macro = rep['macro avg']['f1-score']
        # Compute AUC for multiclass using one-vs-rest
        auc = roc_auc_score(y_test, proba, multi_class='ovr')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"AUC (OVR): {auc:.4f}")
        print("\nFull Classification Report:")
        print(classification_report(y_test, preds, target_names=label.classes_))

    # Show figure
    plt.show()
