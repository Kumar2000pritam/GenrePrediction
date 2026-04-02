import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import f1_score, hamming_loss
from preprocess import (
    handle_missing_values,
    process_duration_features,
    bin_release_year,
    clean_rating_column,
    clean_genres,map_genres
)

from plotting import (
    plot_model_comparison,
    plot_combined_global_shap
)
from artifacts_handler import load_artifacts
from utils import genre_maping,build_features

import warnings
warnings.filterwarnings("ignore")

def compute_metrics(y_true, y_pred):
        return {
            "micro_f1": f1_score(y_true, y_pred, average="micro"),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "hamming_loss": hamming_loss(y_true, y_pred)
        }
def validate_both_models(df, model_artifacts,genre_map):
    """
    End-to-end validation for BOTH models using single artifact object.
    """

    print("\n==============================")
    print(" VALIDATION START")
    print("==============================")

    # =====================================================
    # 1. PREPROCESSING
    # =====================================================
    df = handle_missing_values(df)
    df = process_duration_features(df)
    df = bin_release_year(df)
    df = clean_rating_column(df)
    X = build_features(df, model_artifacts).reset_index(drop=True)
    # =====================================================
    # 2. LOAD FROM SINGLE ARTIFACT OBJECT
    # =====================================================
    logreg_models = model_artifacts["logreg_models"]
    xgb_models = model_artifacts["xgb_models"]

    mlb = model_artifacts["mlb"]

    logreg_threshold = model_artifacts["logreg_threshold"]
    xgb_threshold = model_artifacts["xgb_threshold"]

    feature_cols = model_artifacts["feature_columns"]

    
    # =========================================================
    # 4. GENRE PROCESSING PIPELINE
    # =========================================================
    df = clean_genres(df)
    df = map_genres(df, genre_map)
    # df,genre_counts = filter_genres_by_min_count(df, min_count=50)
    # print(df['genre'].value_counts())
    y_true = mlb.transform(df["genre"])
    # print(y_true.shape)
    # =====================================================
    # 3. TEXT EMBEDDINGS
    # =====================================================
    # if model_artifacts.get("text_model") is not None:
    #     df = add_text_embedding(df, model_artifacts["text_model"])
    
    # =====================================================
    # 4. DROP RAW / INTERMEDIATE COLUMNS
    # =====================================================
    drop_cols = [
        'id', 'type', 'country', 'date_added', 'listed_in', 'platform','combined_text','genre',
        'title','description','director','cast',
        'rating','duration',
        'release_year',
        'duration_num','is_season',
        'movie_duration','num_seasons',
        'movie_duration_bin','season_bin',
        'duration_category','release_year_bin',
        'rating_cleaned'
    ]
    
    X = X.drop(columns=drop_cols, errors='ignore')
    # print(X.columns)
    valid_features = [col for col in feature_cols if col in X.columns]
    print(valid_features)
    # create X using ONLY valid features
    X = X[valid_features].fillna(0).values

    print("\n==============================")
    print("GLOBAL SHAP EXPLANATION")
    print("==============================")
    
    # Convert back to DataFrame for feature names
    X_df = pd.DataFrame(X, columns=valid_features)

    print("\n Logreg model SHAP Explainable")
    # Logistic Regression SHAP
    lr_shap = plot_combined_global_shap(
        logreg_models,
        X_df.values,
        model_artifacts["X_train"],
        feature_names=valid_features,
        model_type="logreg"
    )
    
    # XGBoost SHAP
    print("\n XGboost model SHAP Explainable")
    xgb_shap = plot_combined_global_shap(
        xgb_models,
        X_df.values,
        model_artifacts["X_train"],
        feature_names=valid_features,
        model_type="xgb"
    )
    # =====================================================
    # 5. PREDICTIONS - LOGREG
    # =====================================================
    y_prob_lr = np.zeros((X.shape[0], len(logreg_models)))

    for i, model in enumerate(logreg_models):
        y_prob_lr[:, i] = model.predict_proba(X)[:, 1]

    y_pred_lr = (y_prob_lr >= logreg_threshold).astype(int)

    # =====================================================
    # 6. PREDICTIONS - XGB
    # =====================================================
    y_prob_xgb = np.zeros((X.shape[0], len(xgb_models)))

    for i, model in enumerate(xgb_models):
        y_prob_xgb[:, i] = model.predict_proba(X)[:, 1]

    y_pred_xgb = (y_prob_xgb >= xgb_threshold).astype(int)

    # =====================================================
    # 7. METRICS
    # =====================================================
    
    # print(y_true)
    logreg_metrics = compute_metrics(y_true, y_pred_lr)
    xgb_metrics = compute_metrics(y_true, y_pred_xgb)

    # =====================================================
    # 8. RESULTS
    # =====================================================
    print("\n==============================")
    print(" FINAL COMPARISON")
    print("==============================")

    print("\n Logistic Regression")
    print(logreg_metrics)

    print("\n XGBoost")
    print(xgb_metrics)
    # plot model comparison with ideal value
    plot_model_comparison(logreg_metrics, xgb_metrics)

    return {
        "logreg": logreg_metrics,
        "xgboost": xgb_metrics
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--artifacts", type=str, required=True, help="Artifacts folder path")
    parser.add_argument("--rows", type=int, default=200, help="Number of rows to validate")

    args = parser.parse_args()
    map_genre=genre_maping()
    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_csv(args.data)

    # -------------------------
    # Load artifacts
    # -------------------------
    model_artifacts = load_artifacts(args.artifacts)

    # -------------------------
    # Run validation
    # -------------------------
    results = validate_both_models(
        df.loc[:args.rows, :],
        model_artifacts,
        map_genre
    )

    print("\nDONE")
    print(results)

if __name__ == "__main__":
    main()