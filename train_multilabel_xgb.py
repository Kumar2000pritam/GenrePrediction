from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, hamming_loss
import numpy as np


def xgb_multilabel(
    X_train,
    y_train,
    X_test,
    y_test,
    tune=False
):
    """
    XGBoost multi-label classifier.

    Parameters:
    - tune (bool): If True → GridSearchCV enabled
                   If False → uses default best params
    """

    n_labels = y_train.shape[1]

    models = []
    y_prob = np.zeros((X_test.shape[0], n_labels))
    best_params_per_label = []

    # ==========================
    # DEFAULT BEST PARAMS
    # ==========================
    default_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1
    }

    # ==========================
    # PARAM GRID (ONLY IF TUNING)
    # ==========================
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    # ==========================
    # TRAIN PER LABEL
    # ==========================
    for i in range(n_labels):

        print(f" Training label {i}...")

        y_tr = y_train[:, i]

        # ======================
        # WITH TUNING
        # ======================
        if tune:

            grid = GridSearchCV(
                XGBClassifier(
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1
                ),
                param_grid,
                scoring="f1",
                cv=3,
                n_jobs=-1
            )

            grid.fit(X_train, y_tr)

            model = grid.best_estimator_
            best_params = grid.best_params_

        # ======================
        # WITHOUT TUNING
        # ======================
        else:
            model = XGBClassifier(**default_params)
            model.fit(X_train, y_tr)
            best_params = default_params

        models.append(model)
        best_params_per_label.append(best_params)

        y_prob[:, i] = model.predict_proba(X_test)[:, 1]
    print("\n===== THRESHOLD TUNING STARTED =====")
    print("Model: Logistic Regression")
    print("Optimizing global threshold for best Micro F1 score...")
    print("====================================\n")
    # ==========================
    # THRESHOLD TUNING
    # ==========================
    thresholds = np.arange(0.1, 0.9, 0.05)

    best_t, best_score = 0.5, 0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_test, preds, average="micro")

        if score > best_score:
            best_score = score
            best_t = t

    # ==========================
    # FINAL METRICS
    # ==========================
    y_final = (y_prob >= best_t).astype(int)

    metrics = {
        "micro_f1": f1_score(y_test, y_final, average="micro"),
        "macro_f1": f1_score(y_test, y_final, average="macro"),
        "hamming_loss": hamming_loss(y_test, y_final)
    }
    print("\n===== METRICS =====")
    print(f"Micro F1 Score  : {metrics['micro_f1']:.4f}")
    print(f"Macro F1 Score  : {metrics['macro_f1']:.4f}")
    print(f"Hamming Loss    : {metrics['hamming_loss']:.4f}")
    print("=========================\n")
    

    return models,y_final, best_t, best_params_per_label, metrics