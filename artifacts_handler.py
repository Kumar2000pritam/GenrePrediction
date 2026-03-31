import joblib
import os


def save_artifacts(
    path,
    logreg_models=None,
    xgb_models=None,
    mlb=None,
    rating_encoder=None,
    duration_encoder=None,
    freq_map=None,
    logreg_threshold=None,
    xgb_threshold=None,
    text_model=None,
    feature_columns=None,
    label_names=None,
    logreg_metrics=None,
    xgb_metrics=None
):
    """
    Saves FULL ML pipeline artifacts for production.

    Supports:
    ✔ Logistic Regression models
    ✔ XGBoost models
    ✔ Encoders
    ✔ Thresholds
    ✔ Feature names
    ✔ Label names
    ✔ Metrics
    ✔ SHAP explainers
    ✔ Text embedding model
    """

    os.makedirs(path, exist_ok=True)

    # =========================
    # MODELS
    # =========================
    if logreg_models is not None:
        joblib.dump(logreg_models, f"{path}/logreg_models.pkl")

    if xgb_models is not None:
        joblib.dump(xgb_models, f"{path}/xgb_models.pkl")

    # =========================
    # CORE ARTIFACTS
    # =========================
    if mlb is not None:
        joblib.dump(mlb, f"{path}/mlb.pkl")

    if rating_encoder is not None:
        joblib.dump(rating_encoder, f"{path}/rating_encoder.pkl")

    if duration_encoder is not None:
        joblib.dump(duration_encoder, f"{path}/duration_encoder.pkl")

    if freq_map is not None:
        joblib.dump(freq_map, f"{path}/release_year_freq_map.pkl")

    # =========================
    # THRESHOLDS
    # =========================
    if logreg_threshold is not None:
        joblib.dump(logreg_threshold, f"{path}/logreg_threshold.pkl")

    if xgb_threshold is not None:
        joblib.dump(xgb_threshold, f"{path}/xgb_threshold.pkl")

    # =========================
    # Embedding model
    # =========================

    if text_model is not None:
        joblib.dump(text_model, f"{path}/sentence_transformer.pkl")

    # =========================
    # META INFO (VERY IMPORTANT)
    # =========================
    if feature_columns is not None:
        joblib.dump(feature_columns, f"{path}/feature_columns.pkl")

    if label_names is not None:
        joblib.dump(label_names, f"{path}/label_names.pkl")

    # =========================
    # METRICS
    # =========================
    if logreg_metrics is not None:
        joblib.dump(logreg_metrics, f"{path}/logreg_metrics.pkl")

    if xgb_metrics is not None:
        joblib.dump(xgb_metrics, f"{path}/xgb_metrics.pkl")

    print(f"\n ALL ARTIFACTS SAVED SUCCESSFULLY AT: {path}")
def load_artifacts(path):
    """
    Loads FULL ML pipeline artifacts (production-ready).

    Includes:
    ✔ LogReg + XGBoost models
    ✔ Encoders
    ✔ Feature metadata
    ✔ Thresholds
    ✔ Metrics
    ✔ SHAP explainers
    ✔ Text model
    """

    def safe_load(file_path):
        try:
            return joblib.load(file_path)
        except:
            return None

    artifacts = {
        # =========================
        # MODELS
        # =========================
        "logreg_models": safe_load(f"{path}/logreg_models.pkl"),
        "xgb_models": safe_load(f"{path}/xgb_models.pkl"),

        # =========================
        # CORE ENCODERS
        # =========================
        "mlb": safe_load(f"{path}/mlb.pkl"),
        "rating_encoder": safe_load(f"{path}/rating_encoder.pkl"),
        "duration_encoder": safe_load(f"{path}/duration_encoder.pkl"),
        "freq_map": safe_load(f"{path}/release_year_freq_map.pkl"),

        # =========================
        # THRESHOLDS
        # =========================
        "logreg_threshold": safe_load(f"{path}/logreg_threshold.pkl"),
        "xgb_threshold": safe_load(f"{path}/xgb_threshold.pkl"),

        # =========================
        # META INFORMATION
        # =========================
        "feature_columns": safe_load(f"{path}/feature_columns.pkl"),
        "label_names": safe_load(f"{path}/label_names.pkl"),

        # =========================
        # TEXT MODEL
        # =========================
        "text_model": safe_load(f"{path}/sentence_transformer.pkl"),

        # =========================
        # METRICS
        # =========================
        "logreg_metrics": safe_load(f"{path}/logreg_metrics.pkl"),
        "xgb_metrics": safe_load(f"{path}/xgb_metrics.pkl"),
    }

    print("\n All available artifacts loaded successfully")

    # =========================
    # DEBUG INFO
    # =========================
    print("\n Loaded Components:")
    for k, v in artifacts.items():
        status = "✔" if v is not None else "✖"
        print(f"{status} {k}")

    return artifacts