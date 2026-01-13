import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# Liste des colonnes produits
PRODUCT_COLUMNS = [
    'P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ',
    'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL',
    'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3'
]

# Colonnes catégorielles à encoder
CATEGORICAL_COLUMNS = ['sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code']

# Colonnes numériques à normaliser
NUMERICAL_COLUMNS = ['join_year', 'age', 'total_products']


def preprocess(df):
    """
    Preprocess the input DataFrame by standardizing date formats, extracting year information,
    calculating age, and dropping unnecessary columns.
    :param df: Input DataFrame with columns 'ID', 'join_date', 'birth_year', 'sex', etc.
    :return: Preprocessed DataFrame
    """

    # Convert 'join_date' to datetime, standardize
    df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')

    # Standardize 'sex' to uppercase
    df['sex'] = df['sex'].str.upper()

    # Extract 'join_year' from 'join_date' and calculate 'age'
    df['join_year'] = df['join_date'].dt.year
    current_year = 2020
    df['age'] = current_year - df['birth_year']

    df['total_products'] = df[PRODUCT_COLUMNS].sum(axis=1)

    # Drop unnecessary columns
    cols = list(df.columns)
    cols = [col for col in cols if col not in ['ID', 'join_date', 'birth_year']]
    df = df[cols]

    return df

def prepare_for_training(df, fit_encoders=True, encoder=None, scaler=None):
    """
    Prépare les données pour l'entraînement d'un modèle de classification multi-label en utilisant One-Hot Encoding
    et la normalisation des variables numériques.

    Etapes:
    1. Supprimer la colonne 'Unnamed: 0' si elle est présente.

    2. Gérer les valeurs manquantes:
        - Pour les colonnes catégorielles, remplacer les NaN par 'UNKNOWN'.
        - Pour les colonnes numériques, remplacer les NaN par la médiane de la colonne.

    3. Séparer les colonnes catégorielles et numériques.

    4. Appliquer le One-Hot Encoding aux colonnes catégorielles.

    5. Normaliser les colonnes numériques avec StandardScaler.

    :param df: DataFrame préprocessé
    :param fit_encoders: Si True, ajuste les encoders sur les données
    :param encoder: OneHotEncoder déjà ajusté (pour le mode test)
    :param scaler: Scaler déjà ajusté (pour le mode test)
    :return: X (features), y (labels), encoder, scaler
    """

    df = df.copy()

    # 1. Supprimer 'Unnamed: 0' si présent
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # 2. Gestion des valeurs manquantes
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN').astype(str)

    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 3. Séparer colonnes catégorielles et numériques
    cat_cols_present = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    num_cols_present = [col for col in NUMERICAL_COLUMNS if col in df.columns]

    # 4. One-Hot Encoding des variables catégorielles
    if fit_encoders:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(df[cat_cols_present])
    else:
        cat_encoded = encoder.transform(df[cat_cols_present])

    cat_encoded_df = pd.DataFrame(
        cat_encoded,
        columns=encoder.get_feature_names_out(cat_cols_present),
        index=df.index
    )

    # 5. Normalisation des variables numériques
    if fit_encoders:
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(df[num_cols_present])
    else:
        num_scaled = scaler.transform(df[num_cols_present])

    num_scaled_df = pd.DataFrame(
        num_scaled,
        columns=num_cols_present,
        index=df.index
    )

    # 6. Combiner features
    X = pd.concat([num_scaled_df, cat_encoded_df], axis=1)

    # 7. Extraire labels
    label_columns = [col for col in PRODUCT_COLUMNS if col in df.columns]
    y = df[label_columns] if label_columns else None

    return X, y, encoder, scaler


def split_data(df, test_size=0.2, random_state=42):
    """
    Sépare le DataFrame en ensembles d'entraînement et de test.

    :param df: DataFrame complet
    :param test_size: Proportion des données pour le test (défaut: 0.2)
    :param random_state: Seed pour la reproductibilité (défaut: 42)
    :return: df_train, df_test
    """
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)
