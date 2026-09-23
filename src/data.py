from pathlib import Path

import pandas as pd


def load_city_mapping(mapping_path: str) -> pd.DataFrame:
    """
    Load the geo city cleaning mapping file.
    This is used to clean customer city names.
    """
    path = Path(mapping_path)

    if not path.exists():
        raise FileNotFoundError(f"City mapping file not found: {mapping_path}")

    city_mapping = pd.read_csv(path)[
        [
            "geolocation_zip_code_prefix",
            "geolocation_state",
            "geolocation_city",
            "city_clean",
        ]
    ].drop_duplicates()

    return city_mapping


def apply_city_mapping(df: pd.DataFrame, city_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the city cleaning mapping to the dataframe.
    """
    df = df.copy()

    df = df.merge(
        city_mapping,
        left_on=["customer_zip_code", "customer_state", "customer_city"],
        right_on=[
            "geolocation_zip_code_prefix",
            "geolocation_state",
            "geolocation_city",
        ],
        how="left",
    )

    # Replace original city with cleaned city
    df["customer_city"] = df["city_clean"].fillna(df["customer_city"])

    # Remove mapping columns
    df = df.drop(
        columns=[
            "geolocation_zip_code_prefix",
            "geolocation_state",
            "geolocation_city",
            "city_clean",
        ],
        errors="ignore",
    )

    return df
