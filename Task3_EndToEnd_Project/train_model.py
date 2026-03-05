import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[["hours_studied","attendance","previous_marks"]]
y = data["final_marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "student_model.pkl")

print("Model trained and saved successfully!")