import joblib
import pandas as pd

# Load trained model
model = joblib.load("xgboost_cervical_risk_model.pkl")

print("🩺 Cervical Cancer Risk Prediction")
print("Please answer the following questions:\n")

# ✅ Only keep the questions you want to ask
questions = {
    "Age": "Enter age: ",
    "Number of sexual partners": "Enter number of sexual partners: ",
    "First sexual intercourse": "Enter age at first sexual intercourse: ",
    "Num of pregnancies": "Enter number of pregnancies: ",
    "Smokes": "Do you smoke? (1 = Yes, 0 = No): ",
    "Smokes (years)": "If you smoke, for how many years? (0 if No): ",
    "Smokes (packs/year)": "If you smoke, how many packs per year? (0 if No): ",
    "Hormonal Contraceptives": "Do you use hormonal contraceptives? (1 = Yes, 0 = No): ",
    "Hormonal Contraceptives (years)": "If yes, for how many years? (0 if No): ",
    "STDs:HIV": "Do you have HIV? (1 = Yes, 0 = No): ",
}

# 🧩 Collect user answers
answers = {}
for feature, question in questions.items():
    value = input(question)
    answers[feature] = float(value)

# Convert to DataFrame
user_df = pd.DataFrame([answers])

# ✅ Align with the model’s expected features
expected_cols = model.get_booster().feature_names

# Add any missing columns (fill them with 0)
for col in expected_cols:
    if col not in user_df.columns:
        user_df[col] = 0

# Reorder to match model’s expected feature order
user_df = user_df[expected_cols]

# ✅ Make the prediction
pred = model.predict(user_df)[0] + 1

# Map numerical results back to readable labels
risk_labels = {1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
print(f"\n🧾 Prediction: {risk_labels.get(pred, 'Unknown Risk')}")
