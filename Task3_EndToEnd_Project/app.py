from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("student_model.pkl")

@app.route("/")
def home():
    return "Student Marks Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    hours = data["hours_studied"]
    attendance = data["attendance"]
    previous = data["previous_marks"]

    prediction = model.predict([[hours, attendance, previous]])

    return jsonify({
        "Predicted Marks": prediction[0]
    })

if __name__ == "__main__":
    app.run(debug=True)