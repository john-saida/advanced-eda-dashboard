import pandas as pd
from scipy.stats import zscore

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from eda_analysis import get_numeric_features


def preprocess_data(
    df,
    missing_strategy="none",
    outlier_method="none",
    encoding_method="none",
    scaling_method="none",
):

    
    # Work on a copy so the original DataFrame isn't modified.
    df = df.copy()

    if df.empty:
        raise ValueError("Cannot preprocess an empty dataset.")

    # ==========================================
    # 1. MISSING VALUE TREATMENT
    # ==========================================

    if missing_strategy == "drop":
        df = df.dropna()

    elif missing_strategy in ("mean", "median"):
        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        numeric_df = df[numeric_columns]

        if missing_strategy == "mean":
            fill_values = numeric_df.mean()
        else:
            fill_values = numeric_df.median()

        df[numeric_columns] = numeric_df.fillna(fill_values)

    elif missing_strategy == "mode":
        for column in df.columns:
            mode_values = df[column].mode(dropna=True)

            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values.iloc[0])

    # ==========================================
    # 2. REMEMBER ORIGINAL NUMERIC FEATURES
    # ==========================================

    # Keep track of numerical features before encoding.
    # Categorical encoded columns won't be scaled later.
    original_numeric_columns = list(
        get_numeric_features(df).columns
    )

    # ==========================================
    # 3. OUTLIER TREATMENT
    # ==========================================
    # IQR outlier handling: cap extreme values
    if outlier_method == "iqr":

        numeric_df = get_numeric_features(df)

        if not numeric_df.empty:
            q1 = numeric_df.quantile(0.25)
            q3 = numeric_df.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Limit extreme values instead of deleting rows

            df[numeric_df.columns] = numeric_df.clip(
                lower = lower_bound,
                upper = upper_bound,
                axis=1
            )


            

    elif outlier_method == "zscore":

        numeric_df = get_numeric_features(df)

        if not numeric_df.empty:
            z_scores = numeric_df.apply(
                lambda column: zscore(
                    column, nan_policy="omit"
                )
            )

            # Preserve rows with missing values.
            # Only actual Z-scores above 3 in absolute value
            # cause a row to be removed.
            outlier_mask = (
                z_scores.abs().gt(3).fillna(False)
            ).any(axis=1)

            df = df.loc[~outlier_mask].copy()

    # ==========================================
    # 4. CATEGORICAL ENCODING
    # ==========================================

    categorical_columns = list(
        df.select_dtypes(
            include=["object", "string", "category"]
        ).columns
    )

    if encoding_method == "onehot" and categorical_columns:

        # Represent categorical missing values explicitly.
        for column in categorical_columns:
            df[column] = (
                df[column]
                .astype("object")
                .where(df[column].notna(), "Missing")
            )

        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            dtype=int,
        )

    elif encoding_method == "label" and categorical_columns:

        for column in categorical_columns:
            # Avoid converting missing values to the string "nan".
            values = (
                df[column]
                .astype("object")
                .where(df[column].notna(), "Missing")
                .astype(str)
            )

            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(values)

    # ==========================================
    # 5. FEATURE SCALING
    # ==========================================

    # Scale only the original numerical features.
    # Do not scale one-hot or label-encoded categories.
    numeric_columns = [
        column
        for column in original_numeric_columns
        if column in df.columns
    ]

    if numeric_columns and scaling_method != "none":

        if scaling_method == "standard":
            scaler = StandardScaler()

        elif scaling_method == "minmax":
            scaler = MinMaxScaler()

        elif scaling_method == "robust":
            scaler = RobustScaler()

        else:
            raise ValueError("Invalid feature scaling method.")

        df[numeric_columns] = scaler.fit_transform(
            df[numeric_columns]
        )

    return df
