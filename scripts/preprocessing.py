import pandas as pd


def preprocess_row(df):
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

    # Drop unnecessary columns
    cols = list(df.columns)
    cols = [col for col in cols if col not in ['ID', 'join_date', 'birth_year']]
    df = df[cols]

    return df