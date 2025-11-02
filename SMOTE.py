import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 1️⃣ Load your cleaned dataset
df = pd.read_csv(
    r"C:\Users\jycab\Documents\Thesis\dropped_irrelevant_columns_&_created_multiclassification_column_based_on_ground_truth.csv"
)

# 2️⃣ Check original class distribution
print("Original class distribution:")
print(df["risk_level"].value_counts())

# 3️⃣ Merge very small class (1) with nearby class (2)
df["risk_level"] = df["risk_level"].replace({1: 2})

# Re-check after merging
print("\nClass distribution after merging 1 → 2:")
print(df["risk_level"].value_counts())

# 4️⃣ Remap the labels for readability:
# 0 → 1 (Low Risk)
# 2 → 2 (Medium Risk)
# 3 → 3 (High Risk)
df["risk_level"] = df["risk_level"].replace({0: 1, 2: 2, 3: 3})

print("\nRenamed risk levels for clarity:")
print(df["risk_level"].value_counts().sort_index())
print("\nLegend: 1 = Low, 2 = Medium, 3 = High Risk")

# 5️⃣ Split data into features (X) and target (y)
X = df.drop(columns=["risk_level"])
y = df["risk_level"]

# 6️⃣ Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7️⃣ Apply SMOTE only to the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 8️⃣ Show results before and after balancing
print("\nBefore SMOTE:")
print(y_train.value_counts().sort_index())

print("\nAfter SMOTE:")
print(y_train_res.value_counts().sort_index())

# 9️⃣ Save balanced training data
resampled_df = pd.concat([X_train_res, y_train_res], axis=1)
resampled_df.to_csv(r"C:\Users\jycab\Documents\Thesis\smote_balanced_train_data.csv", index=False)
print("\n✅ Balanced training data saved as 'smote_balanced_train_data.csv'")

# 🔟 Save test set (not oversampled)
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv(r"C:\Users\jycab\Documents\Thesis\original_test_data.csv", index=False)
print("✅ Original (unbalanced) test data saved as 'original_test_data.csv'")

# 🧩 Optional: verify unique labels in each
print("\nFinal Train Classes:", sorted(y_train_res.unique()))
print("Final Test Classes:", sorted(y_test.unique()))
