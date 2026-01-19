import argparse
import pandas as pd

from scripts.preprocessing import preprocess
from scripts.predict import predict, create_submission


def preprocess_and_save(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load raw data, preprocess it, and save to file.

    :param input_path: Path to raw CSV (Train.csv or Test.csv format)
    :param output_path: Path to save preprocessed CSV (train_.csv format)
    :return: Preprocessed DataFrame
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Shape: {df.shape}")

    print("Preprocessing data...")
    df_processed = preprocess(df.copy())
    print(f"  Shape after preprocessing: {df_processed.shape}")

    # Save preprocessed data
    df_processed.to_csv(output_path)
    print(f"Preprocessed data saved to {output_path}")

    return df_processed


def run_prediction_pipeline(
    test_data_path: str,
    model_name: str = "catboost",
    weights_dir: str = "weights",
    output_path: str = "submission.csv"
) -> pd.DataFrame:
    """
    Run the full prediction pipeline.

    :param test_data_path: Path to test data CSV
    :param model_name: Model to use for prediction
    :param weights_dir: Directory containing model weights
    :param output_path: Path to save submission file
    :return: Submission DataFrame
    """
    print("=" * 60)
    print("INSURANCE RECOMMENDATION - PREDICTION PIPELINE")
    print("=" * 60)

    # 1. Load test data
    print(f"\n[1/3] Loading test data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)
    print(f"  Shape: {df_test.shape}")
    print(f"  Columns: {df_test.columns.tolist()}")

    # Store IDs before preprocessing
    ids = df_test['ID'].copy()

    # 2. Make predictions
    print(f"\n[2/3] Making predictions with {model_name} model...")
    predictions = predict(df_test, model_name=model_name, weights_dir=weights_dir)
    print(f"  Predictions shape: {predictions.shape}")

    # 3. Create submission file
    print(f"\n[3/3] Creating submission file...")
    submission_df = create_submission(ids, predictions, output_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return submission_df


def run_full_pipeline(
    train_data_path: str = "data/Train.csv",
    test_data_path: str = "data/Test.csv",
    preprocessed_output: str = "data/eda/train_.csv",
    model_name: str = "catboost",
    weights_dir: str = "weights",
    submission_output: str = "submission.csv"
):
    """
    Run the complete pipeline from raw data to submission.

    Steps:
    1. Preprocess Train.csv -> train_.csv
    2. Load Test.csv and make predictions
    3. Generate submission file

    :param train_data_path: Path to training data
    :param test_data_path: Path to test data
    :param preprocessed_output: Path to save preprocessed training data
    :param model_name: Model to use for prediction
    :param weights_dir: Directory containing model weights
    :param submission_output: Path to save submission file
    """
    print("=" * 60)
    print("INSURANCE RECOMMENDATION - FULL PIPELINE")
    print("=" * 60)

    # Step 1: Preprocess training data
    print("\n" + "-" * 40)
    print("STEP 1: PREPROCESSING TRAINING DATA")
    print("-" * 40)
    df_train_processed = preprocess_and_save(train_data_path, preprocessed_output)

    # Display sample of preprocessed data
    print("\nPreprocessed data sample:")
    print(df_train_processed.head(3))

    # Step 2 & 3: Prediction pipeline
    print("\n" + "-" * 40)
    print("STEP 2 & 3: PREDICTION PIPELINE")
    print("-" * 40)
    submission_df = run_prediction_pipeline(
        test_data_path=test_data_path,
        model_name=model_name,
        weights_dir=weights_dir,
        output_path=submission_output
    )

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Training data preprocessed: {train_data_path} -> {preprocessed_output}")
    print(f"Test data used: {test_data_path}")
    print(f"Model used: {model_name}")
    print(f"Submission file: {submission_output}")
    print(f"Submission shape: {submission_df.shape}")

    return df_train_processed, submission_df


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Insurance Recommendation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (preprocess + predict)
  python main.py --full
  
  # Only preprocess training data
  python main.py --preprocess --input data/Train.csv --output data/eda/train_.csv
  
  # Only run prediction on test data
  python main.py --predict --test data/Test.csv --model catboost --submission submission.csv
  
  # Use a different model
  python main.py --full --model xgboost
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--full", action="store_true",
                           help="Run full pipeline (preprocess + predict)")
    mode_group.add_argument("--preprocess", action="store_true",
                           help="Only preprocess data")
    mode_group.add_argument("--predict", action="store_true",
                           help="Only run prediction")

    # Data paths
    parser.add_argument("--train", default="data/Train.csv",
                       help="Path to training data (default: data/Train.csv)")
    parser.add_argument("--test", default="data/Test.csv",
                       help="Path to test data (default: data/Test.csv)")
    parser.add_argument("--input", default="data/Train.csv",
                       help="Input file for preprocessing (default: data/Train.csv)")
    parser.add_argument("--output", default="data/eda/train_.csv",
                       help="Output file for preprocessing (default: data/eda/train_.csv)")

    # Model options
    parser.add_argument("--model", default="catboost",
                       choices=["catboost", "xgboost", "random_forest", "deep_learning"],
                       help="Model to use for prediction (default: catboost)")
    parser.add_argument("--weights", default="weights",
                       help="Directory containing model weights (default: weights)")

    # Output
    parser.add_argument("--submission", default="submission.csv",
                       help="Path to save submission file (default: submission.csv)")

    args = parser.parse_args()

    # Default to full pipeline if no mode specified
    if not (args.full or args.preprocess or args.predict):
        args.full = True

    if args.preprocess:
        # Only preprocess
        preprocess_and_save(args.input, args.output)

    elif args.predict:
        # Only predict
        run_prediction_pipeline(
            test_data_path=args.test,
            model_name=args.model,
            weights_dir=args.weights,
            output_path=args.submission
        )

    else:  # args.full
        # Full pipeline
        run_full_pipeline(
            train_data_path=args.train,
            test_data_path=args.test,
            preprocessed_output=args.output,
            model_name=args.model,
            weights_dir=args.weights,
            submission_output=args.submission
        )


if __name__ == "__main__":
    main()

