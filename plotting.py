import matplotlib.pyplot as plt
import numpy as np
import shap
import pandas as pd

def plot_model_comparison(logreg_metrics, xgb_metrics):

    metrics = ["micro_f1", "macro_f1", "hamming_loss"]

    # Actual values
    logreg_values = [logreg_metrics[m] for m in metrics]
    xgb_values = [xgb_metrics[m] for m in metrics]

    # Ideal values
    ideal_values = [1, 1, 0]

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure()

    # Bars
    plt.bar(x - width, logreg_values, width, label="LogReg")
    plt.bar(x, xgb_values, width, label="XGBoost")
    plt.bar(x + width, ideal_values, width, label="Ideal")

    # Labels
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Model vs Ideal Performance")
    plt.legend()
    print('Please click cross to view next')
    plt.show(block=True)



def plot_combined_global_shap(models, X, feature_names, model_type="xgb"):
    """
    Combines SHAP importance across all labels into ONE global view
    """

    X_df = pd.DataFrame(X, columns=feature_names)

    all_shap_values = []

    for i, model in enumerate(models):

        print(f"Processing model {i}...")

        # -------------------------------
        # Select Explainer
        # -------------------------------
        if model_type == "logreg":
            explainer = shap.LinearExplainer(model, X_df)
            shap_values = explainer(X_df).values
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)

        # -------------------------------
        # Ensure correct shape
        # -------------------------------
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        all_shap_values.append(np.abs(shap_values))

    # -------------------------------
    # Combine across labels
    # -------------------------------
    combined_shap = np.mean(np.stack(all_shap_values), axis=0)

    # -------------------------------
    # GLOBAL SUMMARY PLOT
    # -------------------------------
    print("\n Combined Global SHAP Summary")
    shap.summary_plot(combined_shap, X_df, show=False)
    print('Please click cross to view next')
    plt.show(block=True)

    # -------------------------------
    # GLOBAL FEATURE IMPORTANCE BAR
    # -------------------------------
    print("\n Combined Feature Importance")
    mean_importance = np.mean(combined_shap, axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_importance
    }).sort_values(by="importance", ascending=False)

    print(importance_df.head(20))

    plt.figure()
    plt.barh(importance_df["feature"][:20][::-1],
             importance_df["importance"][:20][::-1])
    plt.title("Top 20 Features (Combined SHAP Importance)")
    plt.xlabel("Mean |SHAP value|")
    print('Please click cross to view next')
    plt.show(block=True)

    return importance_df
def explain_prediction(models, x_instance, feature_names, predicted_labels, label_names, model_type="xgb"):
    """
    Explain ONLY predicted labels using SHAP waterfall
    """

    import shap
    import matplotlib.pyplot as plt

    # x_instance = x_instance.reshape(1, -1)
    x_instance = pd.DataFrame([x_instance], columns=feature_names)
    for i, model in enumerate(models):

        # Skip labels that are NOT predicted
        if predicted_labels[i] == 0:
            continue

        print("\n====================================")
        print(f" Explaining Label: {label_names[i]}")
        print("====================================")

        # Select explainer
        if model_type == "logreg":
            explainer = shap.LinearExplainer(model, x_instance)
            shap_values = explainer(x_instance)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(x_instance)

        # WATERFALL (MAIN)
        print(" Waterfall Plot")
        shap.plots.waterfall(shap_values[0], show=False)
        print('Please click cross to view next')
        plt.show(block=True)

        #  BAR (TOP FEATURES)
        print(" Top Feature Impact")
        shap.plots.bar(shap_values[0], show=False)
        print('Please click cross to view next')
        plt.show(block=True)