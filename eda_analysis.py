import pandas as pd
import seaborn as sns
from scipy.stats import zscore

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_numeric_features(df):

    numeric_df = df.select_dtypes(include="number")

    if df.empty:
        return numeric_df

    identifier_columns = []

    for column in df.columns:
        column_name = str(column).strip().lower()

        is_identifier =(
            column_name =="id"
            or column_name.endswith("_id")
        )
        if is_identifier:
            unique_ratio =(
                df[column].nunique(dropna=False) / len(df)
            )


            if unique_ratio > 0.9:
              
              identifier_columns.append(column)

    return numeric_df.drop(
        columns = identifier_columns,
        errors="ignore"
    )


    

def generate_eda(df):

    # ==============================
    # Dataset Overview
    # ==============================

    rows = df.shape[0]
    columns = df.shape[1]

    missing_values = df.isnull().sum().sum()
    missing_by_column = df.isnull().sum()


    duplicate_rows = df.duplicated().sum()
    duplicate_data = df[df.duplicated()]


    numeric_df = get_numeric_features(df)

    # ==============================
    # Histogram Visualization
    # ==============================

    chart_paths = []

    for column in numeric_df.columns:

        plt.figure(figsize=(8, 5))

        plt.hist(
            numeric_df[column].dropna(),
            bins=10
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        chart_path = f"static/plots/{column}_histogram.png"

        plt.savefig(chart_path)
        plt.close()

        chart_paths.append(
            f"plots/{column}_histogram.png"
        )


    # ==============================
    # Box Plot Visualization
    # ==============================

    box_charts_paths = []

    for column in numeric_df.columns:

        plt.figure(figsize=(8, 5))

        plt.boxplot(
            numeric_df[column].dropna()
        )

        plt.title(f"Box Plot of {column}")
        plt.ylabel(column)

        plt.tight_layout()

        box_charts_path = (
            f"static/plots/{column}_boxplot.png"
        )

        plt.savefig(box_charts_path)
        plt.close()

        box_charts_paths.append(
            f"plots/{column}_boxplot.png"
        )


    # ==============================
    # Correlation Heatmap
    # ==============================

    correlation_path = None

    if len(numeric_df.columns) >= 2:

        correlation = numeric_df.corr()

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        correlation_path = (
            "plots/correlation_Heatmap.png"
        )

        plt.savefig(
            "static/plots/correlation_Heatmap.png"
        )

        plt.close()


    # ==============================
    # IQR Outlier Detection
    # ==============================

    Q1 = numeric_df.quantile(0.25)

    Q3 = numeric_df.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR

    upper_bound = Q3 + 1.5 * IQR

    outliers = numeric_df[
        (numeric_df < lower_bound) |
        (numeric_df > upper_bound)
    ]

    outlier_counts = outliers.notna().sum()

    outlier_counts = outlier_counts[
        outlier_counts > 0
    ]

    outlier_values = outliers.dropna(how="all")

    # Remove columns that contain no outlier values
    outlier_values = outlier_values.dropna(axis=1, how="all") 


    # ==============================
    # Z-score Outlier Detection
    # ==============================

    z_score_outlier_counts = pd.Series(
        dtype="int64"
    )

    z_score_outlier_values = pd.DataFrame()

    if not numeric_df.empty:

        z_scores = numeric_df.apply(
            lambda column: zscore(
                column,
                nan_policy="omit"
            )
        )

        z_score_outliers = numeric_df[
            z_scores.abs() > 3
        ]

        z_score_outlier_counts = (
            z_score_outliers.notna().sum()
        )

        z_score_outlier_counts = (
            z_score_outlier_counts[
                z_score_outlier_counts > 0
            ]
        )

        z_score_outlier_values = (
            z_score_outliers.dropna(
                how="all"
            )
        )


    # ==============================
    # Statistical Analysis
    # ==============================

    numeric_df = get_numeric_features(df)

    mean_values = numeric_df.mean()
    median_values = numeric_df.median()
    min_values = numeric_df.min()
    max_values = numeric_df.max()
    std_values = numeric_df.std()

    if len(numeric_df.columns) > 0:
        summary =numeric_df.describe()
    else:
        summary = pd.DataFrame()

    
    # Skewness and Kurtosis
    skewness = numeric_df.skew()
    kurtosis = numeric_df.kurt()
    data_types = df.dtypes

   # ==========================================
   # CATEGORICAL ANALYSIS
   # ==========================================

    categorical_df = df.select_dtypes(
        include=["object","string","category"]
    )

    categorical_summary = {}

    for column in categorical_df.columns:
        mode_values = categorical_df[column].mode(dropna=True)

        if mode_values.empty:
            most_frequent = None
        else:
            most_frequent = mode_values.iloc[0]


        categorical_summary[column] = {
            "unique_values": categorical_df[column].nunique(),
            "most_frequent": most_frequent,
            "frequency": categorical_df[column].value_counts().to_dict()
        }
      
        # ==========================================
        # UNIQUE VALUE ANALYSIS
        # ==========================================

    unique_value_counts = df.nunique(dropna=False)

        # High-cardinality categorical analysis
    high_cardinality = {}

    for column in categorical_df.columns:

        unique_count = categorical_df[column].nunique(dropna=False)

        if unique_count > 10:
            high_cardinality[column] = unique_count

       
        # ==========================================
        # CATEGORICAL VISUALIZATION
        # ==========================================

    categorical_chart_paths = []

    for column in categorical_df.columns:

       #count non-missing values inn the column
       value_counts = categorical_df[column].value_counts()

       #skip chart generation if thre are no valid values
       if value_counts.empty:
           continue

       plt.figure(figsize=(8, 5))

       value_counts.plot(kind="bar")

       

       plt.title(f"Distribution of {column}")
       plt.xlabel(column)
       plt.ylabel("Frequency")

       plt.xticks(rotation=45)
       plt.tight_layout()

       categorical_chart_path = (
          f"static/plots/{column}_bar_chart.png"
    )

       plt.savefig(categorical_chart_path)
       plt.close()

       categorical_chart_paths.append(
          f"plots/{column}_bar_chart.png"
    )


       
# ==========================================
# SCATTER PLOT ANALYSIS
# ==========================================

    scatter_chart_paths = []
    correlation_pairs = []

    if len(numeric_df.columns) >= 2:

    # 1. Calculate correlation matrix
      correlation_matrix = numeric_df.corr()

    # 2. Store unique column pairs
      


      for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):


            column1 = correlation_matrix.columns[i]
            column2 = correlation_matrix.columns[j]


            correlation_value = correlation_matrix.iloc[i, j]


            # Skip undefined correlations
            if pd.notna(correlation_value):
                correlation_pairs.append(
                    (column1, column2, abs(correlation_value))
                )


    # 3. Sort by strongest correlation
    correlation_pairs.sort(
        key=lambda x: x[2],
        reverse=True
    )

    # 4. Generate at most five scatter plots
    for column1, column2, _ in correlation_pairs[:5]:
          
          

          plt.figure(figsize=(8, 5))


          # Keep only rows where both values are available
          plot_data = df[[column1,column2]].dropna()

          # Skip the plot if no valid rows remain
          if plot_data.empty:
              plt.close()
              continue

          plt.scatter(
            plot_data[column1],
            plot_data[column2],
            alpha=0.5
        )
          

          plt.title(f"{column1} vs {column2}")
          plt.xlabel(column1)
          plt.ylabel(column2)
          plt.tight_layout()


          scatter_chart_path = (
            f"static/plots/{column1}_vs_{column2}_scatter.png"
        )
          

          plt.savefig(scatter_chart_path)
          plt.close()


          scatter_chart_paths.append(
            f"plots/{column1}_vs_{column2}_scatter.png"
        )


       




    
    # ==============================
    # Overview Dictionary
    # ==============================

    overview = {

        "rows": rows,

        "columns": columns,

        "missing_values": missing_values,

        "duplicate_rows": duplicate_rows

    }


    # ==============================
    # Return All EDA Results
    # ==============================

    return {

        "overview": overview,

        "mean_values": mean_values,

        "median_values": median_values,

        "min_values": min_values,

        "max_values": max_values,

        "std_values": std_values,

        "summary": summary,

        "skewness": skewness,

        "kurtosis": kurtosis,

        "categorical_summary": categorical_summary,

        "categorical_chart_paths": categorical_chart_paths,

        "scatter_chart_paths": scatter_chart_paths,

        "unique_value_counts": unique_value_counts,

        "high_cardinality": high_cardinality,

        "data_types": data_types,

        "missing_by_column": missing_by_column,

        "duplicate_data": duplicate_data,

        "Q1": Q1,

        "Q3": Q3,

        "IQR": IQR,

        "lower_bound": lower_bound,

        "upper_bound": upper_bound,

        "outlier_counts": outlier_counts,

        "outlier_values": outlier_values,

        "z_score_outlier_counts":
            z_score_outlier_counts,

        "z_score_outlier_values":
            z_score_outlier_values,

        "chart_paths": chart_paths,

        "box_charts_paths":
            box_charts_paths,

        "correlation_path":
            correlation_path

    }

