from flask import Flask, render_template, request,send_file
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from eda_analysis import generate_eda,get_numeric_features
from preprocessing import preprocess_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/analyze", methods=["POST"])
def analyze():

    # Get uploaded file
    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    # Read CSV
    df = pd.read_csv(file)

    

    # Save original dataset
    file_path = "uploads/" + file.filename
    df.to_csv(file_path, index=False)

    # Generate EDA
    results = generate_eda(df)

    # Display dashboard
    return render_template(
        "index.html",

        overview=results["overview"],

        mean_values=results["mean_values"],
        median_values=results["median_values"],
        min_values=results["min_values"],
        max_values=results["max_values"],
        std_values=results["std_values"],

        summary=results["summary"],
        skewness=results["skewness"],
        kurtosis=results["kurtosis"],
        categorical_summary=results["categorical_summary"],
        categorical_chart_paths=results["categorical_chart_paths"],
        scatter_chart_paths=results["scatter_chart_paths"],
        unique_value_counts=results["unique_value_counts"],
        high_cardinality=results["high_cardinality"],
        data_types=results["data_types"],

        missing_by_column=results["missing_by_column"],
        duplicate_data=results["duplicate_data"],

        Q1=results["Q1"],
        Q3=results["Q3"],
        IQR=results["IQR"],
        lower_bound=results["lower_bound"],
        upper_bound=results["upper_bound"],

        outlier_counts=results["outlier_counts"],
        outlier_values=results["outlier_values"],

        z_score_outlier_counts=
            results["z_score_outlier_counts"],

        z_score_outlier_values=
            results["z_score_outlier_values"],

        chart_paths=results["chart_paths"],

        box_charts_paths=
            results["box_charts_paths"],

        correlation_path=
            results["correlation_path"],

        filename=file.filename
    )
# @app.route("/clean", methods=["POST"])
# def clean():

#     # Get the cleaning option selected by the user
#     strategy = request.form["missing_strategy"]

#     filename = request.form["filename"]
    
#     # Get the name of the uploaded CSV
#     file_path = "uploads/" + request.form["filename"]

#     # Read the saved CSV into Pandas
#     df = pd.read_csv(file_path)

#     # Remove rows containing missing values
#     if strategy == "drop":

#         df = df.dropna()

#     # Fill numerical missing values with mean
#     elif strategy == "mean":

#         numeric_columns = df.select_dtypes(include="number").columns

#         df[numeric_columns] = df[numeric_columns].fillna(
#             df[numeric_columns].mean()
#         )

#     # Fill numerical missing values with median
#     elif strategy == "median":

#         numeric_columns = df.select_dtypes(include="number").columns

#         df[numeric_columns] = df[numeric_columns].fillna(
#             df[numeric_columns].median()
#         )

#     # Fill missing values with mode
#     elif strategy == "mode":

#         for column in df.columns:

#             df[column] = df[column].fillna(
#                 df[column].mode()[0]
#             )
#      #saved cleaned dataset
#     cleaned_file_path = f"uploads/cleaned_{filename}"
#     df.to_csv(cleaned_file_path,index=False)

#     # Convert cleaned DataFrame into an HTML table
#     return render_template(
#     "index.html",
#     cleaned_data=df.to_html(index=False),
#     filename=filename,
#     cleaned_filename =f"cleaned_{filename}"
# )

# @app.route("/handle_outliers",methods=["post"])
# def handle_outliers():
#     # Get the selected outlier method
#     outlier_method = request.form["outlier_method"]

#     # Get the original filename
#     filename = request.form["filename"]

#     # Path of the uploaded dataset
#     file_path = "uploads/" + filename

#     #read the dataset
#     df = pd.read_csv(file_path)

#     #select numerical columns
#     numeric_df = df.select_dtypes(include="number")

#     #keep all rows
#     if outlier_method == "none":
#         cleaned_df = df

#     #remove IQR outliers
#     elif outlier_method == "iqr":
#         Q1 = numeric_df.quantile(0.25)
#         Q3 = numeric_df.quantile(0.75)

#         IQR = Q3 - Q1

#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR

#         mask = ~(
#             (numeric_df < lower_bound)
#             (numeric_df > upper_bound)
#         ).any(axis=1)
#         cleaned_df =df[mask]
#     #remove Z-score outliers
#     elif outlier_method =="zscore":
#         z_score = numeric_df.apply(zscore)
#         mask = (z_score.abs()<=3).all(axis=1)

#         cleaned_df = df[mask]
    
#     # Save the cleaned dataset
#     cleaned_file_path = f"uploads/outlier_cleaned_{filename}"

#     cleaned_df.to_csv(
#         cleaned_file_path,
#         index=False
#     )

#     # Display cleaned dataset
#     return render_template(
#         "index.html",
#         cleaned_data=cleaned_df.to_html(index=False),
#         filename=filename,
#         cleaned_filename=f"outlier_cleaned_{filename}"
#     )



@app.route("/preprocess", methods=["POST"])
def preprocess():


    
    # Get the selected dataset
    filename = request.form["filename"]

    # Get preprocessing options
    missing_strategy = request.form.get(
        "missing_strategy", "none"
    )
    outlier_method = request.form.get(
        "outlier_method", "none"
    )
    encoding_method = request.form.get(
        "encoding_method", "none"
    )
    scaling_method = request.form.get(
        "scaling_method", "none"
    )

    # Read the original dataset
    file_path = "uploads/" + filename
    df = pd.read_csv(file_path)

    # Run preprocessing
    df = preprocess_data(
        df,
        missing_strategy=missing_strategy,
        outlier_method=outlier_method,
        encoding_method=encoding_method,
        scaling_method=scaling_method,
    )

    

    # Save the processed dataset
    processed_filename = "processed_" + filename
    processed_file_path = "uploads/" + processed_filename

    df.to_csv(processed_file_path, index=False)

    # Generate EDA for the processed dataset
    results = generate_eda(df)

    # Render the dashboard
    return render_template(
        "index.html",
        overview=results["overview"],
        categorical_summary=results["categorical_summary"],
        categorical_chart_paths=results["categorical_chart_paths"],
        scatter_chart_paths=results["scatter_chart_paths"],
        mean_values=results["mean_values"],
        median_values=results["median_values"],
        min_values=results["min_values"],
        max_values=results["max_values"],
        std_values=results["std_values"],
        summary=results["summary"],
        skewness=results["skewness"],
        kurtosis=results["kurtosis"],
        unique_value_counts=results["unique_value_counts"],
        high_cardinality=results["high_cardinality"],
        data_types=results["data_types"],
        missing_by_column=results["missing_by_column"],
        duplicate_data=results["duplicate_data"],
        Q1=results["Q1"],
        Q3=results["Q3"],
        IQR=results["IQR"],
        lower_bound=results["lower_bound"],
        upper_bound=results["upper_bound"],
        outlier_counts=results["outlier_counts"],
        outlier_values=results["outlier_values"],
        z_score_outlier_counts=results["z_score_outlier_counts"],
        z_score_outlier_values=results["z_score_outlier_values"],
        chart_paths=results["chart_paths"],
        box_charts_paths=results["box_charts_paths"],
        correlation_path=results["correlation_path"],
        filename=processed_filename,
        current_filename=processed_filename,
        cleaned_data=df.head(100).to_html(index=False),
        cleaned_filename=processed_filename,
    )


@app.route("/download/<filename>")
def download(filename):
    file_path = "uploads/" + filename
    return send_file(file_path,as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)