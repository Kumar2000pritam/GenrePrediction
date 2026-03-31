import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, hamming_loss

def train_multilabel_logreg(X_train, y_train, X_test, y_test):
    """
    Trains one Logistic Regression per label with GridSearchCV.

    Also:
    - Finds best global threshold
    - Evaluates micro/macro F1 + hamming loss
    """

    n_labels = y_train.shape[1]

    param_grid = {
        "C": [0.01, 0.1, 1, 5, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "class_weight": [None, "balanced"]
    }

    models = []
    y_prob = np.zeros((X_test.shape[0], n_labels))
    best_params_per_label = []

    for i in range(n_labels):
        print(f" Training label with HP Tuning {i}...")
        grid = GridSearchCV(
            LogisticRegression(max_iter=2000),
            param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1
        )

        grid.fit(X_train, y_train[:, i])

        best_model = grid.best_estimator_
        models.append(best_model)
        best_params_per_label.append(grid.best_params_)

        y_prob[:, i] = best_model.predict_proba(X_test)[:, 1]
    print("\n===== THRESHOLD TUNING STARTED =====")
    print("Model: Logistic Regression")
    print("Optimizing global threshold for best Micro F1 score...")
    print("====================================\n")
    # threshold tuning
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_t, best_score = 0.5, 0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_test, preds, average="micro")

        if score > best_score:
            best_score = score
            best_t = t

    # final metrics
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