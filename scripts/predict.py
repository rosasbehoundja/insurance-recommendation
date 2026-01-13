import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

# Conditional import for tensorflow (not always needed)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

from preprocessing import preprocess, prepare_for_training, PRODUCT_COLUMNS


def load_sklearn_model(model_path: Union[str, Path]):
    """
    Load a sklearn-based model (RandomForest, XGBoost, CatBoost) from pickle file.

    :param model_path: Path to the .pkl file
    :return: model, encoder, scaler
    """
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    return data['model'], data['encoder'], data['scaler']


def load_deep_learning_model(model_path: Union[str, Path]):
    """
    Load a Keras deep learning model.

    :param model_path: Path to the .keras file
    :return: Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for deep learning model. Install with: pip install tensorflow")
    return tf.keras.models.load_model(str(model_path))


def predict_with_sklearn(model, encoder, scaler, df: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using a sklearn-based model.

    :param model: Trained sklearn model
    :param encoder: Fitted OneHotEncoder
    :param scaler: Fitted StandardScaler
    :param df: Preprocessed DataFrame (after preprocess() function)
    :return: Predictions array (n_samples, n_products)
    """
    # Prepare features using existing encoder and scaler
    X, _, _, _ = prepare_for_training(df, fit_encoders=False, encoder=encoder, scaler=scaler)

    # Make predictions
    predictions = model.predict(X)

    return predictions


def predict_with_deep_learning(model, encoder, scaler, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """
    Make predictions using the deep learning model.

    :param model: Trained Keras model
    :param encoder: Fitted OneHotEncoder
    :param scaler: Fitted StandardScaler
    :param df: Preprocessed DataFrame (after preprocess() function)
    :param threshold: Threshold for converting probabilities to binary predictions
    :return: Predictions array (n_samples, n_products)
    """
    # Prepare features using existing encoder and scaler
    X, _, _, _ = prepare_for_training(df, fit_encoders=False, encoder=encoder, scaler=scaler)

    # Convert to numpy array
    X_np = X.values.astype(np.float32)

    # Get probabilities
    y_proba = model.predict(X_np)

    # Apply threshold
    predictions = (y_proba >= threshold).astype(int)

    return predictions


def predict(df: pd.DataFrame, model_name: str = "catboost", weights_dir: str = "weights") -> np.ndarray:
    """
    Main prediction function that handles preprocessing and model loading.

    :param df: Raw DataFrame (from Train.csv or Test.csv format)
    :param model_name: Model to use ('xgboost', 'catboost', 'random_forest', 'deep_learning')
    :param weights_dir: Directory containing model weights
    :return: Predictions array (n_samples, n_products)
    """
    weights_path = Path(weights_dir)

    # Preprocess the data
    df_processed = preprocess(df.copy())

    if model_name == "deep_learning":
        # Load deep learning model
        model = load_deep_learning_model(weights_path / "deep_learning.keras")
        # Load encoder and scaler from another model (they're the same)
        _, encoder, scaler = load_sklearn_model(weights_path / "catboost.pkl")
        predictions = predict_with_deep_learning(model, encoder, scaler, df_processed)
    else:
        # Load sklearn-based model
        model_file = weights_path / f"{model_name}.pkl"
        model, encoder, scaler = load_sklearn_model(model_file)
        predictions = predict_with_sklearn(model, encoder, scaler, df_processed)

    return predictions


def create_submission(ids: pd.Series, predictions: np.ndarray, output_path: str = "submission.csv"):
    """
    Create a submission file in the format required (ID X PCODE, Label).

    :param ids: Series of IDs from the test set
    :param predictions: Predictions array (n_samples, n_products)
    :param output_path: Path to save the submission CSV
    :return: DataFrame with submission format
    """
    submissions = []

    for idx, row_id in enumerate(ids):
        for prod_idx, product in enumerate(PRODUCT_COLUMNS):
            submissions.append({
                'ID X PCODE': f"{row_id} X {product}",
                'Label': int(predictions[idx, prod_idx])
            })

    submission_df = pd.DataFrame(submissions)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Total rows: {len(submission_df)}")

    return submission_df

