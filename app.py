import pickle
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from preprocessing import preprocess, PRODUCT_COLUMNS, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

app = Flask(__name__)

VARS_LIST = [
    'join_year', 'age', 'total_products', 'sex_F', 'sex_M', 'marital_status_D',
    'marital_status_M', 'marital_status_P', 'marital_status_R', 'marital_status_S',
    'marital_status_U', 'marital_status_W', 'marital_status_f', 'branch_code_1X1H',
    'branch_code_30H5', 'branch_code_49BM', 'branch_code_748L', 'branch_code_94KC',
    'branch_code_9F9T', 'branch_code_BOAS', 'branch_code_E5SW', 'branch_code_EU3L',
    'branch_code_O4JC', 'branch_code_O67J', 'branch_code_UAOD', 'branch_code_X23B',
    'branch_code_XX25', 'branch_code_ZFER', 'occupation_code_00MO', 'occupation_code_0B60',
    'occupation_code_0KID', 'occupation_code_0OJM', 'occupation_code_0PO7', 'occupation_code_0S50',
    'occupation_code_0SH6', 'occupation_code_0VYC', 'occupation_code_1AN5', 'occupation_code_1DT6',
    'occupation_code_1H8Y', 'occupation_code_1MB4', 'occupation_code_1MSV', 'occupation_code_1NFK',
    'occupation_code_1YKL', 'occupation_code_2346', 'occupation_code_2686', 'occupation_code_2A7I',
    'occupation_code_2BE6', 'occupation_code_2G86', 'occupation_code_2HLT', 'occupation_code_2JHV',
    'occupation_code_2MBB', 'occupation_code_2R78', 'occupation_code_2US6', 'occupation_code_2XZ1',
    'occupation_code_2YAO', 'occupation_code_31GG', 'occupation_code_31JW', 'occupation_code_374O',
    'occupation_code_3NHZ', 'occupation_code_3X46', 'occupation_code_3YQ1', 'occupation_code_44SU',
    'occupation_code_4M0E', 'occupation_code_4W0D', 'occupation_code_59QM', 'occupation_code_5FPK',
    'occupation_code_5JRZ', 'occupation_code_5LNN', 'occupation_code_5OVC', 'occupation_code_6E4H',
    'occupation_code_6KYM', 'occupation_code_6LKA', 'occupation_code_6PE7', 'occupation_code_6SKY',
    'occupation_code_6XXU', 'occupation_code_6YZA', 'occupation_code_734F', 'occupation_code_738L',
    'occupation_code_73AC', 'occupation_code_74BF', 'occupation_code_7G9M', 'occupation_code_7KM4',
    'occupation_code_7UDQ', 'occupation_code_7UHW', 'occupation_code_7UWC', 'occupation_code_820B',
    'occupation_code_834U', 'occupation_code_8HRZ', 'occupation_code_8Y24', 'occupation_code_9B5B',
    'occupation_code_9FA1', 'occupation_code_9HD1', 'occupation_code_9IM8', 'occupation_code_9IP9',
    'occupation_code_A4ZC', 'occupation_code_A793', 'occupation_code_AIDS', 'occupation_code_AIIN',
    'occupation_code_APO0', 'occupation_code_AQIB', 'occupation_code_B3QW', 'occupation_code_B8W8',
    'occupation_code_BER4', 'occupation_code_BFD1', 'occupation_code_BP09', 'occupation_code_BPSA',
    'occupation_code_BWBW', 'occupation_code_BX9E', 'occupation_code_C1E3', 'occupation_code_C8F6',
    'occupation_code_CAAV', 'occupation_code_CEL6', 'occupation_code_CV2C', 'occupation_code_CYDC',
    'occupation_code_DD8W', 'occupation_code_DE5D', 'occupation_code_DHSN', 'occupation_code_DPRV',
    'occupation_code_DZRV', 'occupation_code_E39I', 'occupation_code_E5PF', 'occupation_code_EE5R',
    'occupation_code_F35Z', 'occupation_code_F57O', 'occupation_code_FJBW', 'occupation_code_FLNZ',
    'occupation_code_FLXH', 'occupation_code_FSWO', 'occupation_code_FSXG', 'occupation_code_GQ0N',
    'occupation_code_GVZ1', 'occupation_code_GWEP', 'occupation_code_GZA8', 'occupation_code_H1K7',
    'occupation_code_HAXM', 'occupation_code_HJF4', 'occupation_code_HSVE', 'occupation_code_HTQS',
    'occupation_code_I2OD', 'occupation_code_I31I', 'occupation_code_IE90', 'occupation_code_IJ01',
    'occupation_code_IMHI', 'occupation_code_INEJ', 'occupation_code_IQFS', 'occupation_code_IUT9',
    'occupation_code_IX8T', 'occupation_code_IZ77', 'occupation_code_J9SY', 'occupation_code_JHU5',
    'occupation_code_JI64', 'occupation_code_JN20', 'occupation_code_JQH3', 'occupation_code_JS7M',
    'occupation_code_JSAX', 'occupation_code_JUIP', 'occupation_code_K0DL', 'occupation_code_K5GV',
    'occupation_code_K5LB', 'occupation_code_KBWO', 'occupation_code_KNVN', 'occupation_code_KPG9',
    'occupation_code_KUPK', 'occupation_code_L1P3', 'occupation_code_L4PL', 'occupation_code_LAYD',
    'occupation_code_LGTN', 'occupation_code_LLLH', 'occupation_code_LQ0W', 'occupation_code_M0WG',
    'occupation_code_MEFQ', 'occupation_code_MU16', 'occupation_code_N2ZZ', 'occupation_code_N7K2',
    'occupation_code_NDL9', 'occupation_code_NFJH', 'occupation_code_NO3L', 'occupation_code_NQW1',
    'occupation_code_NSJX', 'occupation_code_NX5Y', 'occupation_code_OEH6', 'occupation_code_OME4',
    'occupation_code_ONY7', 'occupation_code_OPVX', 'occupation_code_OQMY', 'occupation_code_OYQF',
    'occupation_code_P2K2', 'occupation_code_P4MD', 'occupation_code_PJR4', 'occupation_code_PKW3',
    'occupation_code_PMAI', 'occupation_code_PPNK', 'occupation_code_PSUY', 'occupation_code_PWCW',
    'occupation_code_Q0LY', 'occupation_code_Q231', 'occupation_code_Q2L0', 'occupation_code_Q57T',
    'occupation_code_Q6J6', 'occupation_code_QJID', 'occupation_code_QQUP', 'occupation_code_QQVA',
    'occupation_code_QS0L', 'occupation_code_QX54', 'occupation_code_QZYX', 'occupation_code_R44Q',
    'occupation_code_R7GL', 'occupation_code_RE69', 'occupation_code_RF6M', 'occupation_code_RH2K',
    'occupation_code_RM3L', 'occupation_code_RSN9', 'occupation_code_RUFT', 'occupation_code_RXV3',
    'occupation_code_RY9B', 'occupation_code_S96O', 'occupation_code_S9KU', 'occupation_code_SF1X',
    'occupation_code_SS6D', 'occupation_code_SST3', 'occupation_code_SSTX', 'occupation_code_T6AB',
    'occupation_code_TUN1', 'occupation_code_U37O', 'occupation_code_U9RX', 'occupation_code_UBBX',
    'occupation_code_UC7E', 'occupation_code_UJ5T', 'occupation_code_URYD', 'occupation_code_UYDZ',
    'occupation_code_V4XX', 'occupation_code_VREH', 'occupation_code_VVTC', 'occupation_code_VYSA',
    'occupation_code_VZN9', 'occupation_code_W1X2', 'occupation_code_W3Y9', 'occupation_code_W3ZV',
    'occupation_code_WE0G', 'occupation_code_WE7U', 'occupation_code_WIWP', 'occupation_code_WMTK',
    'occupation_code_WSID', 'occupation_code_WSRG', 'occupation_code_WV7U', 'occupation_code_WVQF',
    'occupation_code_X1JO', 'occupation_code_XC1N', 'occupation_code_XHJD', 'occupation_code_XVMH',
    'occupation_code_Y1WG', 'occupation_code_Y7G1', 'occupation_code_YJXM', 'occupation_code_YMGT',
    'occupation_code_YX47', 'occupation_code_Z7PM', 'occupation_code_ZA1S', 'occupation_code_ZCQR',
    'occupation_code_ZHC2', 'occupation_code_ZKQ3', 'occupation_code_ZWPL', 'occupation_category_code_56SI',
    'occupation_category_code_90QI', 'occupation_category_code_AHH5', 'occupation_category_code_JD7X',
    'occupation_category_code_L44T', 'occupation_category_code_T4MS'
]

WEIGHTS_DIR = Path(__file__).parent / "weights"

def load_sklearn_model(model_path):
    """Load a sklearn-based model from pickle file."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoder'], data['scaler']


def load_deep_learning_model(model_path):
    """Load a Keras deep learning model."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for deep learning model.")
    return tf.keras.models.load_model(str(model_path))


def format_for_prediction(df_preprocessed, encoder, scaler):
    """
    Format the preprocessed DataFrame to match the expected feature columns.
    Uses the encoder and scaler from training to ensure consistency.

    :param df_preprocessed: DataFrame after preprocess() function
    :param encoder: Fitted OneHotEncoder from training
    :param scaler: Fitted StandardScaler from training
    :return: DataFrame with correctly formatted features
    """
    df = df_preprocessed.copy()

    # Handle missing values
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN').astype(str)

    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Get present columns
    cat_cols_present = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    num_cols_present = [col for col in NUMERICAL_COLUMNS if col in df.columns]

    # One-Hot Encoding
    cat_encoded = encoder.transform(df[cat_cols_present])
    cat_encoded_df = pd.DataFrame(
        cat_encoded,
        columns=encoder.get_feature_names_out(cat_cols_present),
        index=df.index
    )

    # Scale numerical columns
    num_scaled = scaler.transform(df[num_cols_present])
    num_scaled_df = pd.DataFrame(
        num_scaled,
        columns=num_cols_present,
        index=df.index
    )

    # Combine features (same order as training)
    X = pd.concat([num_scaled_df, cat_encoded_df], axis=1)

    return X


def make_prediction(df, model_name="xgboost"):
    """
    Make predictions using the specified model.

    :param df: Raw input DataFrame
    :param model_name: 'xgboost' or 'deep_learning'
    :return: Dictionary with product predictions
    """
    # Add product columns with 0 values (required for preprocess)
    for col in PRODUCT_COLUMNS:
        df[col] = 0

    # Preprocess the data
    df_processed = preprocess(df.copy())

    # Load model and scaler
    if model_name == "deep_learning":
        model = load_deep_learning_model(WEIGHTS_DIR / "deep_learning.keras")
        _, encoder, scaler = load_sklearn_model(WEIGHTS_DIR / "xgboost.pkl")
    else:
        model, encoder, scaler = load_sklearn_model(WEIGHTS_DIR / "xgboost.pkl")

    # Format data for prediction
    X = format_for_prediction(df_processed, encoder, scaler)

    # Make predictions
    if model_name == "deep_learning":
        X_np = X.values.astype(np.float32)
        y_proba = model.predict(X_np)
        predictions = (y_proba >= 0.5).astype(int)
    else:
        predictions = model.predict(X)

    # Format results
    results = {}
    for i, product in enumerate(PRODUCT_COLUMNS):
        results[product] = int(predictions[0][i]) if predictions.ndim > 1 else int(predictions[i])

    return results


@app.route('/')
def index():
    """Render the main form page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get form data
        data = {
            'ID': [request.form.get('id', 'USER001')],
            'join_date': [request.form.get('join_date')],
            'sex': [request.form.get('sex')],
            'marital_status': [request.form.get('marital_status')],
            'birth_year': [int(request.form.get('birth_year'))],
            'branch_code': [request.form.get('branch_code')],
            'occupation_code': [request.form.get('occupation_code')],
            'occupation_category_code': [request.form.get('occupation_category_code')]
        }

        model_name = request.form.get('model', 'xgboost')

        # Create DataFrame
        df = pd.DataFrame(data)

        # Make prediction
        predictions = make_prediction(df, model_name)

        # Get recommended products (where prediction = 1)
        recommended = [product for product, pred in predictions.items() if pred == 1]

        return render_template('result.html',
                             predictions=predictions,
                             recommended=recommended,
                             model_name=model_name,
                             user_data=data)

    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    try:
        data = request.get_json()

        df = pd.DataFrame({
            'ID': [data.get('id', 'API_USER')],
            'join_date': [data.get('join_date')],
            'sex': [data.get('sex')],
            'marital_status': [data.get('marital_status')],
            'birth_year': [int(data.get('birth_year'))],
            'branch_code': [data.get('branch_code')],
            'occupation_code': [data.get('occupation_code')],
            'occupation_category_code': [data.get('occupation_category_code')]
        })

        model_name = data.get('model', 'xgboost')
        predictions = make_prediction(df, model_name)

        recommended = [product for product, pred in predictions.items() if pred == 1]

        return jsonify({
            'success': True,
            'model': model_name,
            'predictions': predictions,
            'recommended_products': recommended
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

