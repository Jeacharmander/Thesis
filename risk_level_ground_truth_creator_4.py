# =========================================
# 1. Import Libraries
# =========================================
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# =========================================
# 2. Load Dataset
# =========================================
# Replace this with your actual dataset path
df = pd.read_csv(r'C:\Users\jycab\Documents\Thesis\dropped_columns_according_to_doc.csv')

# =========================================
# 3. Define Risk Assessment Function
# =========================================
def assess_risk(row):
    score = 0

    # HPV and HIV
    if row.get('STDs:HPV', 0) == 1:
        score += 3
    if row.get('STDs:HIV', 0) == 1:
        score += 3

    # Smoking
    if row.get('Smokes', 0) == 1:
        score += 2
        if pd.notnull(row.get('Smokes (years)', None)) and row['Smokes (years)'] > 5:
            score += 1

    # Hormonal Contraceptives
    if pd.notnull(row.get('Hormonal Contraceptives (years)', None)) and row['Hormonal Contraceptives (years)'] >= 5:
        score += 2

    # Number of pregnancies
    preg = row.get('Num of pregnancies', 0)
    if pd.notnull(preg):
        if preg >= 4:
            score += 2
        elif 2 <= preg <= 3:
            score += 1

    # Number of sexual partners
    partners = row.get('Number of sexual partners', 0)
    if pd.notnull(partners):
        if partners >= 4:
            score += 2
        elif 2 <= partners <= 3:
            score += 1

    # Final classification
    if score <= 2:
        return 0  # Low
    elif score <= 5:
        return 1  # Medium
    else:
        return 2  # High

# =========================================
# 4. Create Risk Level Column
# =========================================
df['risk_level'] = df.apply(assess_risk, axis=1)

print("Original class counts:")
print(df['risk_level'].value_counts())

# =========================================
# 5. Select Relevant Columns
# =========================================
features = [
    'Number of sexual partners',
    'Num of pregnancies',
    'Smokes',
    'Smokes (years)',
    'Smokes (packs/year)',
    'Hormonal Contraceptives (years)',
    'STDs:HIV',
    'STDs:HPV'
]

X = df[features]
y = df['risk_level']

# =========================================
# 6. Split Train/Test
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================================
# 7. Apply SMOTE to BOTH Train and Test
# =========================================
sm = SMOTE(random_state=42)

X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
X_test_bal, y_test_bal = sm.fit_resample(X_test, y_test)

# =========================================
# 8. Combine Back into DataFrames
# =========================================
df_train_balanced = pd.DataFrame(X_train_bal, columns=features)
df_train_balanced['risk_level'] = y_train_bal

df_test_balanced = pd.DataFrame(X_test_bal, columns=features)
df_test_balanced['risk_level'] = y_test_bal

# Merge for a full balanced dataset (optional)
df_balanced_all = pd.concat([df_train_balanced, df_test_balanced], ignore_index=True)

# =========================================
# 9. Verify Class Counts
# =========================================
print("\nBalanced TRAIN class counts:")
print(df_train_balanced['risk_level'].value_counts())

print("\nBalanced TEST class counts:")
print(df_test_balanced['risk_level'].value_counts())

# =========================================
# 10. Save Files
# =========================================
df_train_balanced.to_csv(r'C:\Users\jycab\Documents\Thesis\Datasets\balanced_train.csv', index=False)
df_test_balanced.to_csv(r'C:\Users\jycab\Documents\Thesis\Datasets\balanced_test.csv', index=False)
df_balanced_all.to_csv(r'C:\Users\jycab\Documents\Thesis\Datasets\balanced_full_dataset.csv', index=False)

print("\n✅ Balanced datasets saved as:")
print(" - balanced_train.csv")
print(" - balanced_test.csv")
print(" - balanced_full_dataset.csv")
