import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, accuracy_score
)

# XGBoost & CatBoost
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Preprocessing local
from preprocessing import (
    preprocess, prepare_for_training, split_data, PRODUCT_COLUMNS
)


RANDOM_STATE = 42
MODELS_DIR = Path("../weights")
MODELS_DIR.mkdir(exist_ok=True)


def evaluate_multilabel(y_true, y_pred, model_name="Model"):
    """
    Évalue les performances d'un modèle de classification multi-label.
    """
    results = {
        "model": model_name,
        "f1_micro": f1_score(y_true, y_pred, average='micro'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
        "precision_micro": precision_score(y_true, y_pred, average='micro'),
        "recall_micro": recall_score(y_true, y_pred, average='micro'),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
    }

    print(f"\n{'='*50}")
    print(f"Résultats pour {model_name}")
    print(f"{'='*50}")
    print(f"F1 Score (micro):     {results['f1_micro']:.4f}")
    print(f"F1 Score (macro):     {results['f1_macro']:.4f}")
    print(f"F1 Score (weighted):  {results['f1_weighted']:.4f}")
    print(f"Precision (micro):    {results['precision_micro']:.4f}")
    print(f"Recall (micro):       {results['recall_micro']:.4f}")
    print(f"Hamming Loss:         {results['hamming_loss']:.4f}")
    print(f"Subset Accuracy:      {results['subset_accuracy']:.4f}")

    return results


def save_model(model, name, encoder=None, scaler=None):
    """Sauvegarde un modèle et ses transformers."""
    model_path = MODELS_DIR / f"{name}.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'encoder': encoder,
            'scaler': scaler
        }, f)

    print(f"✅ Modèle sauvegardé: {model_path}")

# MODÈLE 1: RANDOM FOREST


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Entraîne un Random Forest pour classification multi-label.
    """
    print("\n Entraînement Random Forest...")

    rf_base = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )

    model = MultiOutputClassifier(rf_base, n_jobs=-1)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluation
    results = evaluate_multilabel(y_test, y_pred, "Random Forest")

    return model, results


# MODÈLE 2: XGBOOST

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Entraîne un XGBoost pour classification multi-label.
    """
    print("\n Entraînement XGBoost...")

    xgb_base = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model = MultiOutputClassifier(xgb_base, n_jobs=-1)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluation
    results = evaluate_multilabel(y_test, y_pred, "XGBoost")

    return model, results


# MODÈLE 3: CATBOOST

def train_catboost(X_train, y_train, X_test, y_test):
    """
    Entraîne un CatBoost pour classification multi-label.
    """
    print("\n Entraînement CatBoost...")

    cb_base = CatBoostClassifier(
        iterations=200,
        depth=8,
        learning_rate=0.1,
        random_seed=RANDOM_STATE,
        verbose=False,
        auto_class_weights='Balanced'
    )

    model = MultiOutputClassifier(cb_base, n_jobs=-1)
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Évaluation
    results = evaluate_multilabel(y_test, y_pred, "CatBoost")

    return model, results


# MODÈLE 4: DEEP LEARNING (TENSORFLOW)

def build_deep_model(input_dim, output_dim):
    """
    Construit un réseau de neurones pour classification multi-label.
    """
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dropout(0.2),

        # Couche de sortie: sigmoid pour multi-label
        Dense(output_dim, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def train_deep_learning(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de Deep Learning pour classification multi-label.
    """
    print("\n Entraînement Deep Learning (TensorFlow)...")

    # Conversion en arrays numpy
    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_train_np = y_train.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.float32)

    # Construction du modèle
    model = build_deep_model(X_train_np.shape[1], y_train_np.shape[1])

    print(model.summary())

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Entraînement
    history = model.fit(
        X_train_np, y_train_np,
        validation_data=(X_test_np, y_test_np),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Prédictions (seuil à 0.5)
    y_pred_proba = model.predict(X_test_np)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Évaluation
    results = evaluate_multilabel(y_test_np, y_pred, "Deep Learning (TensorFlow)")

    return model, history, results


# PIPELINE PRINCIPAL

def load_and_prepare_data(data_path="../data/Train.csv"):
    """
    Charge et prépare les données pour l'entraînement.
    """
    print("Chargement des données...")
    df = pd.read_csv(data_path)

    print("Préprocessing...")
    df = preprocess(df)

    # Calculer total_products
    df['total_products'] = df[PRODUCT_COLUMNS].sum(axis=1)

    print("Split train/test...")
    df_train, df_test = split_data(df, test_size=0.2, random_state=RANDOM_STATE)

    print("Préparation pour l'entraînement...")
    X_train, y_train, encoder, scaler = prepare_for_training(df_train, fit_encoders=True)
    X_test, y_test, _, _ = prepare_for_training(df_test, fit_encoders=False, encoder=encoder, scaler=scaler)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test, encoder, scaler


def main():
    """
    Fonction principale pour entraîner tous les modèles.
    """
    print("="*60)
    print(" ZIMNAT INSURANCE - CLASSIFICATION MULTI-LABEL")
    print("="*60)

    # Charger et préparer les données
    X_train, y_train, X_test, y_test, encoder, scaler = load_and_prepare_data()

    # Stockage des résultats
    all_results = []

    # 1. Random Forest
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    save_model(rf_model, "random_forest", encoder, scaler)
    all_results.append(rf_results)

    # 2. XGBoost
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    save_model(xgb_model, "xgboost", encoder, scaler)
    all_results.append(xgb_results)

    # 3. CatBoost
    cb_model, cb_results = train_catboost(X_train, y_train, X_test, y_test)
    save_model(cb_model, "catboost", encoder, scaler)
    all_results.append(cb_results)

    # 4. Deep Learning
    dl_model, dl_history, dl_results = train_deep_learning(X_train, y_train, X_test, y_test)
    dl_model.save(MODELS_DIR / "deep_learning.keras")
    print(f"Modèle Deep Learning sauvegardé: {MODELS_DIR / 'deep_learning.keras'}")
    all_results.append(dl_results)

    # Résumé comparatif
    print("\n" + "="*60)
    print("RÉSUMÉ COMPARATIF DES MODÈLES")
    print("="*60)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index('model')
    print(results_df.round(4).to_string())

    # Meilleur modèle selon F1-micro
    best_model = results_df['f1_micro'].idxmax()
    print(f"\n Meilleur modèle (F1-micro): {best_model}")

    # Sauvegarder les résultats
    results_df.to_csv(MODELS_DIR / "model_comparison.csv")
    print(f"Résultats sauvegardés: {MODELS_DIR / 'model_comparison.csv'}")

    return results_df


if __name__ == "__main__":
    main()

