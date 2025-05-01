from utils import (
    load_data,
    prepare_data,
    scale_features,
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    save_model,
)
from catboost import CatBoostClassifier
import os


def main():
    try:
        # Step 1: Load the resampled multi-class dataset
        resampled_data_path = "../data/resampled_column_3C_weka.csv"
        resampled_data = load_data(resampled_data_path)

        # Step 2: Prepare features and target
        X, y = prepare_data(resampled_data)

        # Step 3: Split the resampled data into training and testing sets
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Step 4: Scale the features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # Step 5: Train the CatBoost model
        model = CatBoostClassifier(
            allow_writing_files=False,
            logging_level="Silent",
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # Step 6: Save the trained model
        save_model(model, "../models/CatBoost_multi.pkl")

        # Step 7: Evaluate the model
        accuracy, report, metrics = evaluate_model(
            model, X_test_scaled, y_test, task_type="multi"
        )
        print(f"Model Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        print("Additional Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Step 8: Visualize the confusion matrix
        reports_dir = "../reports/"
        os.makedirs(reports_dir, exist_ok=True)
        y_pred = model.predict(X_test_scaled)
        plot_confusion_matrix(
            y_test,
            y_pred,
            class_labels=["Normal", "Disk Hernia", "Spondylolisthesis"],
            title="Confusion Matrix - CatBoost",
            save_path=os.path.join(reports_dir, "confusion_matrix_catboost.png"),
        )

        # Step 9: Plot feature importance
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(
                feature_names=X.columns,
                importances=model.feature_importances_,
                title="Feature Importance - CatBoost",
                save_path=os.path.join(reports_dir, "feature_importance_catboost.png"),
            )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
