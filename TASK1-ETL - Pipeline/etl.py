# ===============================
# DATA PIPELINE DEVELOPMENT - TASK 1
# CODTECH DATA SCIENCE INTERNSHIP
# ===============================

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

print("Loading dataset...")

# EXTRACT
df = pd.read_csv("data.csv")
print("Dataset loaded successfully.\n")

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
categorical_features = df.select_dtypes(include=["object"]).columns

# TRANSFORM

# Numeric Pipeline
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical Pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

print("Applying transformations...")

processed_data = preprocessor.fit_transform(df)

# Convert to DataFrame
processed_df = pd.DataFrame(
    processed_data.toarray() if hasattr(processed_data, "toarray") else processed_data
)

# LOAD
processed_df.to_csv("processed_data.csv", index=False)

print("Transformation Completed")
print("Processed data saved as processed_data.csv")
print("ETL Process Completed Successfully!")
