import numpy as np
import argparse
import pandas as pd
from plotting import explain_prediction
from preprocess import (
    process_duration_features,
    bin_release_year,
    clean_rating_column
)
from utils import build_features
from artifacts_handler import load_artifacts


def inference_model(df, artifacts):
    df = process_duration_features(df)
    df = bin_release_year(df)
    df = clean_rating_column(df)

    logreg_models = artifacts["logreg_models"]
    xgb_models = artifacts["xgb_models"]
    mlb = artifacts["mlb"]
    feature_names=artifacts["feature_columns"]
    logreg_threshold = artifacts["logreg_threshold"]
    xgb_threshold = artifacts["xgb_threshold"]

    # -------------------------
    # Features
    # -------------------------
    X = build_features(df, artifacts).reset_index(drop=True)

    n_samples = X.shape[0]
    n_labels = len(xgb_models)

    logreg_prob = np.zeros((n_samples, n_labels))
    xgb_prob = np.zeros((n_samples, n_labels))

    # -------------------------
    # Logistic Regression
    # -------------------------
    for i, model in enumerate(logreg_models):
        proba = model.predict_proba(X)
        logreg_prob[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba.ravel()

    # -------------------------
    # XGBoost
    # -------------------------
    for i, model in enumerate(xgb_models):
        proba = model.predict_proba(X)
        xgb_prob[:, i] = proba[:, 1] if proba.shape[1] == 2 else proba.ravel()

    # -------------------------
    # Apply separate thresholds
    # -------------------------
    logreg_pred = (logreg_prob >= logreg_threshold).astype(int)
    xgb_pred = (xgb_prob >= xgb_threshold).astype(int)

    
    # print(X.values)
    explain_prediction(
    models=xgb_models,
    x_instance=X.values[0],
    X_train=artifacts["X_train"],
    feature_names=feature_names,
    predicted_labels=xgb_pred[0],
    label_names=mlb.classes_,
    model_type="xgb"
    )
    explain_prediction(
    models=logreg_models,
    x_instance=X.values[0],
    X_train=artifacts["X_train"],
    feature_names=feature_names,
    predicted_labels=logreg_pred[0],
    label_names=mlb.classes_,
    model_type="logreg"
    )
    # -------------------------
    # Ensemble (vote)
    # -------------------------
    y_pred = ((logreg_pred + xgb_pred) >= 1).astype(int)

    # -------------------------
    # Convert to labels
    # -------------------------
    logreg_labels = mlb.inverse_transform(logreg_pred)
    xgb_labels = mlb.inverse_transform(xgb_pred)
    final_labels = mlb.inverse_transform(y_pred)
    print(final_labels)
    return {
        "logreg_pred": logreg_pred,
        "xgb_pred": xgb_pred,
        "ensemble_pred": y_pred,
    
        "logreg_labels": logreg_labels,
        "xgb_labels": xgb_labels,
        "predicted_labels": final_labels,
    
        "logreg_prob": logreg_prob,
        "xgb_prob": xgb_prob
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on one record")
    parser.add_argument( "--artifacts_path",type=str,required=True,help="Path to trained model artifacts (e.g., ml_artifacts)")
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--duration", type=str, required=True)
    parser.add_argument("--release_year", type=int, required=True)
    parser.add_argument("--rating", type=str, required=True)

    # additional fields
    parser.add_argument("--director", type=str, default="")
    parser.add_argument("--cast", type=str, default="")
    parser.add_argument("--country", type=str, default="")
    parser.add_argument("--date_added", type=str, default="")
    parser.add_argument("--platform", type=str, default="")

    return parser.parse_args()
def main():
    args = parse_args()

    # load artifacts dynamically
    artifacts = load_artifacts(args.artifacts_path)

    df = pd.DataFrame([{
        "title": args.title,
        "director": args.director,
        "cast": args.cast,
        "description": args.description,
        "duration": args.duration,
        "release_year": args.release_year,
        "rating": args.rating
    }])

    results = inference_model(df, artifacts)

    print(results["predicted_labels"][0])
    print("\n===== RESULTS =====")
    
    print("\n Probabilities (XGB):")
    print(results["xgb_prob"][0])

    print("\n Probabilities (LogReg):")
    print(results["logreg_prob"][0])
    print("\n Final Predicted Labels:")
    print(results["predicted_labels"][0])

    print("\n Logistic Regression Labels:")
    print(results["logreg_labels"][0])

    print("\n XGBoost Labels:")
    print(results["xgb_labels"][0])

    

if __name__ == "__main__":
    main()