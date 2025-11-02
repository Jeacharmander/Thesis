import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv(r"K:\Jilliana Abogado\021125\balanced_risk_dataset_downsampled.csv")

print(df["risk_level"].value_counts())


# Remove the ground truth in the train so that it will learn all the features.
X = df.drop(columns=["risk_level"])  # all features
y = df["risk_level"]  # target column


# Split into train + temp (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

# Split temp into validation and test equally
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print("Train:", y_train.value_counts())
print("Val:", y_val.value_counts())
print("Test:", y_test.value_counts())

X_train.assign(risk_level=y_train).to_csv(r"K:\Jilliana Abogado\021125\train.csv", index=False)
X_val.assign(risk_level=y_val).to_csv(r"K:\Jilliana Abogado\021125\val.csv", index=False)
X_test.assign(risk_level=y_test).to_csv(r"K:\Jilliana Abogado\021125\test.csv", index=False)
