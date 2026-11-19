# Advanced EDA Dashboard

A Flask web application for exploring CSV datasets through statistical analysis, data-quality checks, visualizations, and preprocessing options.

## Features

- Upload a CSV dataset and view an analysis dashboard
- Dataset overview: row count, column count, missing values, and duplicate rows
- Numerical summaries: mean, median, minimum, maximum, standard deviation, skewness, and kurtosis
- Categorical summaries, value frequencies, unique-value counts, and high-cardinality checks
- Visualizations including histograms, boxplots, categorical charts, scatter plots, and a correlation heatmap
- Outlier analysis using IQR and Z-score methods
- Data preprocessing options for missing values, outliers, categorical encoding, and feature scaling
- View and download the processed dataset

## Tech stack

Python, Flask, Pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, and HTML/Jinja templates.

## Project structure

```text
advanced-eda-dashboard/
├── app.py
├── eda_analysis.py
├── preprocessing.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── plots/
│       └── .gitkeep
└── uploads/
    └── .gitkeep
```

The `static/plots/` and `uploads/` directories are used by the application at runtime. Generated charts and uploaded CSV files are intentionally not included in this repository.

## Run locally

1. Install Python.
2. Open a terminal in this project directory.
3. (Recommended) Create and activate a virtual environment.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   python app.py
   ```

6. Open the local address printed in the terminal (typically `http://127.0.0.1:5000`) and upload a CSV file.

## Project status

The project has been tested locally by its author. Further improvements and security hardening may be made in future updates.

## Notes

- Do not commit private, confidential, or personal datasets.
- This repository is intended for local demonstration. Before deploying the application publicly, add upload validation, safe filename/path handling, appropriate error handling, and disable Flask debug mode.
