import numpy as np
import shap
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss
from preprocess import (
    handle_missing_values,
    create_combined_text,
    bin_release_year,
    clean_genres,
    map_genres,
    filter_genres_by_min_count,
    process_duration_features,
    clean_rating_column
)
from encoding import (
    encode_rating,
    frequency_encode_release_year,
    encode_duration_category,
    generate_text_embeddings
)
from X_y_split import split_data

from utils import (
    merge_tabular_and_text,
    final_cleanup
)
from train_multilabel_logreg import train_multilabel_logreg
from train_multilabel_xgb import xgb_multilabel
from plotting import (
    plot_model_comparison,
    plot_combined_global_shap
)

def run_full_ml_pipeline(df, genre_map, min_genre_count=50):
    """
    END-TO-END ML PIPELINE (SELF-CONTAINED)

    This version:
    ✔ Creates y internally (MultiLabelBinarizer)
    ✔ Runs full preprocessing
    ✔ Trains multi-label model
    ✔ Builds explainability artifacts

    Returns:
    dict of trained models + encoders + metrics
    """
    print("\n==============================")
    print(" DATA PREPROCESSING STARTED")
    print("==============================")
    # =========================================================
    # 1. MISSING VALUE HANDLING
    # =========================================================
    df = handle_missing_values(df)
    # =========================================================
    # 2. TEXT FEATURE ENGINEERING
    # =========================================================
    df = create_combined_text(df)
    # =========================================================
    # 3. RELEASE YEAR BINNING
    # =========================================================
    df = bin_release_year(df)

   
    # =========================================================
    # 4. GENRE PROCESSING PIPELINE
    # =========================================================
    df = clean_genres(df)
   
    df = map_genres(df, genre_map)
    df,genre_counts = filter_genres_by_min_count(df, min_count=50)
    # =========================================================
    # 5. DURATION FEATURES
    # =========================================================
    df = process_duration_features(df)
    
    # =========================================================
    # 6. CREATE TARGET (y) INTERNALLY
    # =========================================================
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['genre'])

    # store label names
    label_names = mlb.classes_

    # =========================================================
    # 7. CLEAN RATING COLUMN
    # =========================================================
    df = clean_rating_column(df)

    # =========================================================
    # 8. DROP LEAKAGE / RAW COLUMNS
    # =========================================================
    cols_to_drop = [
        'id', 'type', 'title', 'director', 'cast', 'country', 'date_added',
        'release_year', 'duration', 'listed_in', 'description', 'rating',
        'platform', 'duration_num', 'country_grouped', 'genre',
        'is_season', 'movie_duration', 'num_seasons',
        'movie_duration_bin', 'season_bin', 'text_input'
    ]

    df = df.drop(columns=cols_to_drop, errors='ignore')
    print("\n==============================")
    print(" TRAIN-TEST SPLIT")
    print("==============================")
    # =========================================================
    # 9. TRAIN-TEST SPLIT
    # =========================================================
    
    X_train, X_test, y_train, y_test = split_data(
        df, y,
        test_size=0.2,
        random_state=42
    )
    # print(X_train.isnull().sum())
    # print(X_test.isnull().sum())
    print("\n==============================")
    print(" ENCODING STARTED")
    print("==============================")
    # =========================================================
    # 10. ENCODINGS
    # =========================================================
    X_train, X_test, oe_rating = encode_rating(X_train, X_test)
    X_train, X_test, freq_map = frequency_encode_release_year(X_train, X_test)
    X_train, X_test, oe_duration = encode_duration_category(X_train, X_test)

    print("\n==============================")
    print(" TEXT - TITLE, DESCRIPTION, DIRECTOR, CAST EMBEDDING STARTED")
    print("==============================")
    # =========================================================
    # 11. TEXT EMBEDDINGS
    # =========================================================
    train_emb, test_emb, text_model = generate_text_embeddings(X_train, X_test)

    # =========================================================
    # 12. MERGE FEATURES
    # =========================================================

    X_train, X_test = merge_tabular_and_text(
        X_train, X_test,
        train_emb, test_emb
    )

    # =========================================================
    # 13. FINAL CLEANUP
    # =========================================================
    final_drop_cols = [
        'combined_text', 'release_year_bin',
        'director', 'cast', 'country', 'date_added',
        'release_year', 'rating', 'duration', 'listed_in',
        'description', 'platform', 'duration_num',
        'is_season', 'movie_duration', 'num_seasons',
        'movie_duration_bin', 'season_bin',
        'duration_category', 'country_grouped',
        'text_input', 'rating_cleaned'
    ]

    X_train, X_test = final_cleanup(X_train, X_test, final_drop_cols)

   
    X_train = pd.DataFrame(X_train)
    X_train.columns = X_train.columns.astype(str)
    X_test = pd.DataFrame(X_test)
    X_test.columns = X_test.columns.astype(str)
    
    # =========================================================
    # 14. MODEL TRAINING
    # =========================================================
    print("\n==============================")
    print(" TRAINING LOGISTIC REGRESSION MODEL")
    print("==============================")
    
    logreg_models, y_final_logreg,logreg_threshold, logreg_params, logreg_metrics = train_multilabel_logreg(
        X_train, y_train, X_test, y_test
    )
    
    print("\n==============================")
    print(" TRAINING XGBOOST MODEL MODEL")
    print("==============================")
    
    xgb_models,y_final_xgb, xgb_threshold, xgb_params, xgb_metrics = xgb_multilabel(
        X_train,
        y_train,
        X_test,
        y_test,
        tune=False
    )
    y_ensemble = ((y_final_logreg + y_final_xgb) >= 1).astype(int)
    final_metrics={"micro_f1": f1_score(y_test, y_ensemble, average="micro"),
    "macro_f1": f1_score(y_test, y_ensemble, average="macro"),
    "hamming_loss": hamming_loss(y_test, y_ensemble)
    }
    print("\n===== FINAL COMBINED MODEL PERFORMANCE =====")
    print(f"Micro F1 Score  : {final_metrics['micro_f1']:.4f}")
    print(f"Macro F1 Score  : {final_metrics['macro_f1']:.4f}")
    print(f"Hamming Loss    : {final_metrics['hamming_loss']:.4f}")
    print("===================================\n")

    print("\n===== MODEL EXPLANATION =====")
    print("EXPLAINABILITY INSIGHTS GENERATED SUCCESSFULLY")
    print("===================================\n")
    plot_combined_global_shap(xgb_models, X_test,X_train, list(X_train.columns), model_type="xgb")
    plot_combined_global_shap(logreg_models, X_test, X_train,list(X_train.columns), model_type="logreg")
    
    plot_model_comparison(xgb_metrics=xgb_metrics,logreg_metrics=logreg_metrics)
    # =========================================================
    # 16. RETURN FULL ARTIFACT PACKAGE
    # =========================================================
    return {
    # =========================
    # MODELS
    # =========================
    "logreg_models": logreg_models,
    "xgb_models": xgb_models,

    # =========================
    # THRESHOLDS
    # =========================
    "logreg_threshold": logreg_threshold,
    "xgb_threshold": xgb_threshold,

    # =========================
    # METRICS
    # =========================
    "logreg_metrics": logreg_metrics,
    "xgb_metrics": xgb_metrics,

    # =========================
    # BEST PARAMS
    # =========================
    "logreg_best_params": logreg_params,
    "xgb_best_params": xgb_params,

    # =========================
    # FEATURE INFO
    # =========================
    "feature_columns": list(X_train.columns),
    "label_names": label_names,

    # =========================
    # ENCODERS / ARTIFACTS
    # =========================
    "mlb": mlb,
    "rating_encoder": oe_rating,
    "duration_encoder": oe_duration,
    "freq_map": freq_map,
    "text_model": text_model,
    # =========================
    # DATA
    # =========================
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}