import joblib
import pandas as pd
from pathlib import Path


def load_preprocessor(preprocessor_path: str):
    """
    Load the fitted preprocessor (ColumnTransformer).
    Never fit it again — only load the saved object.
    """
    path = Path(preprocessor_path)

    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    preprocessor = joblib.load(path)
    return preprocessor


def apply_preprocessor(df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """
    Apply the already-fitted preprocessor to the dataframe.
    Returns a DataFrame with the exact same feature names as training.
    """
    # Transform
    processed = preprocessor.transform(df)

    # Get the feature names that were created during training
    feature_names = preprocessor.get_feature_names_out()

    # Convert back to DataFrame
    processed_df = pd.DataFrame(
        processed,
        columns=feature_names,
        index=df.index
    )

    return processed_df