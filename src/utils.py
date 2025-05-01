import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib
import seaborn as sns
import matplotlib.pyplot as plt


# Data Loading
def load_data(file_path):
    """Load dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


# Feature Engineering
def prepare_data(df):
    """Separate features and target from the dataset."""
    X = df.drop(columns=["Target"])
    y = df["Target"]
    return X, y


def scale_features(X_train, X_test):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# Model Handling
def save_model(model, file_path):
    """Save the trained model to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)


# Model Evaluation
def evaluate_model(model, X_test, y_test, task_type="binary"):
    """
    Evaluates the model and returns accuracy, classification report, and additional metrics.

    Parameters:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        task_type: 'binary' or 'multi' for classification type.

    Returns:
        accuracy: Accuracy score.
        report: Classification report.
        metrics: Dictionary of additional metrics (precision, recall, F1-score, ROC-AUC).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    average_type = "binary" if task_type == "binary" else "weighted"
    precision = precision_score(y_test, y_pred, average=average_type, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average_type, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_type, zero_division=0)

    if y_proba is not None:
        if task_type == "binary":
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    else:
        roc_auc = float("nan")

    metrics = {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
    }

    return accuracy, report, metrics


# Visualization Utilities
def plot_confusion_matrix(
    y_true, y_pred, class_labels, title="Confusion Matrix", cmap="Blues", save_path=None
):
    """Plot a confusion matrix using seaborn's heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def plot_feature_importance(
    feature_names, importances, title="Feature Importance", save_path=None
):
    """Plot feature importance for a model."""
    sorted_indices = importances.argsort()[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=sorted_importances,
        y=sorted_features,
        palette="viridis",
        hue=sorted_features,
        legend=False,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
