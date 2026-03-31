import pandas as pd
from utils import genre_maping
from mlpipeline import run_full_ml_pipeline
from artifacts_handler import save_artifacts, load_artifacts
import argparse
import warnings
warnings.filterwarnings("ignore")
def main():
    parser = argparse.ArgumentParser(description="Train ML Pipeline")

    # -------------------------
    # Arguments
    # -------------------------
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="ml_artifacts",
        help="Path to save artifacts"
    )

    args = parser.parse_args()

    print("\n==============================")
    print(" PROCESS STARTED")
    print("==============================")

    # -------------------------
    # Load Data
    # -------------------------
    df = pd.read_csv(args.data)

    # -------------------------
    # Genre Mapping
    # -------------------------
    genre_map = genre_maping()

    # -------------------------
    # Run Full Pipeline
    # -------------------------
    artifacts = run_full_ml_pipeline(
        df=df,
        genre_map=genre_map
    )

    # -------------------------
    # Save Artifacts
    # -------------------------
    save_artifacts(
        path=args.out,

        logreg_models=artifacts["logreg_models"],
        xgb_models=artifacts["xgb_models"],

        mlb=artifacts["mlb"],
        rating_encoder=artifacts["rating_encoder"],
        duration_encoder=artifacts["duration_encoder"],
        freq_map=artifacts["freq_map"],

        logreg_threshold=artifacts["logreg_threshold"],
        xgb_threshold=artifacts["xgb_threshold"],

        text_model=artifacts["text_model"],

        feature_columns=artifacts["feature_columns"],
        label_names=artifacts["label_names"],

        logreg_metrics=artifacts["logreg_metrics"],
        xgb_metrics=artifacts["xgb_metrics"],
    )

    print("\n Artifacts saved successfully!")

    # -------------------------
    # Optional: Reload check
    # -------------------------
    model_artifacts = load_artifacts(args.out)

    print("\n Reload successful. Training pipeline complete.")

if __name__ == "__main__":
    main()