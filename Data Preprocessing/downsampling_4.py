import pandas as pd

df = pd.read_csv(r"K:\Jilliana Abogado\021125\balanced_risk_dataset_with_empty_values_handled.csv")


# Check current class distribution
print("\nBefore downsampling:")
print(df["risk_level"].value_counts())

# Target count (smallest class size)
target_count = df["risk_level"].value_counts().min()

# Downsample each class to target_count
df_balanced = (
    df.groupby("risk_level", group_keys=False)
    .apply(lambda x: x.sample(n=target_count, random_state=42))
    .reset_index(drop=True)
)

# Verify new balanced class distribution
print("\nAfter downsampling:")
print(df_balanced["risk_level"].value_counts())

# Optionally, save the balanced dataset
df_balanced.to_csv(r"K:\Jilliana Abogado\021125\balanced_risk_dataset_downsampled.csv", index=False)
print("\nBalanced dataset saved successfully.")
