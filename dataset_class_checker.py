import pandas as pd

df = pd.read_csv(
    r"C:\Users\jycab\Documents\Thesis\dropped_irrelevant_columns_&_created_multiclassification_column_based_on_ground_truth.csv"
)

print(df["risk_level"].value_counts())
