import sys

import joblib
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QComboBox, QFormLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
                             QWidget)


class CervicalCancerPredictor(QWidget):
    def __init__(self):
        super().__init__()

        # Load trained model
        self.model = joblib.load("xgboost_cervical_risk_model.pkl")

        # Window setup
        self.setWindowTitle("🩺 Cervical Cancer Risk Prediction")
        self.setGeometry(100, 100, 400, 600)

        # Create layout
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Define questions (input fields)
        self.inputs = {}

        questions = {
            "Age": QLineEdit(),
            "Number of sexual partners": QLineEdit(),
            "First sexual intercourse": QLineEdit(),
            "Num of pregnancies": QLineEdit(),
            "Smokes": self.create_combo(),
            "Smokes (years)": QLineEdit(),
            "Smokes (packs/year)": QLineEdit(),
            "Hormonal Contraceptives": self.create_combo(),
            "Hormonal Contraceptives (years)": QLineEdit(),
            "STDs:HIV": self.create_combo(),
        }

        # Add fields to the form
        for label, widget in questions.items():
            form_layout.addRow(QLabel(label + ":"), widget)
            self.inputs[label] = widget

        layout.addLayout(form_layout)

        # Predict button
        predict_btn = QPushButton("🔍 Predict Risk")
        predict_btn.clicked.connect(self.predict_risk)
        layout.addWidget(predict_btn)

        self.setLayout(layout)

    def create_combo(self):
        """Creates a Yes/No dropdown"""
        combo = QComboBox()
        combo.addItems(["0 - No", "1 - Yes"])
        return combo

    def predict_risk(self):
        """Runs when Predict button is clicked"""
        try:
            # Gather input values
            answers = {}
            for key, widget in self.inputs.items():
                if isinstance(widget, QComboBox):
                    answers[key] = float(widget.currentText().split(" - ")[0])
                else:
                    answers[key] = float(widget.text())

            # Convert to DataFrame
            user_df = pd.DataFrame([answers])

            # Align with model columns
            expected_cols = self.model.get_booster().feature_names
            for col in expected_cols:
                if col not in user_df.columns:
                    user_df[col] = 0
            user_df = user_df[expected_cols]

            # Predict
            pred = self.model.predict(user_df)[0] + 1
            risk_labels = {1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
            result = risk_labels.get(pred, "Unknown Risk")

            # Show result
            QMessageBox.information(self, "Prediction Result", f"🧾 Predicted Risk Level: {result}")

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please fill all fields with valid numbers.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CervicalCancerPredictor()
    window.show()
    sys.exit(app.exec())
