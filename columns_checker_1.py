import pandas as pd


def get_original_dataset():
    df = pd.read_csv(r'C:\Cus\Jilliana Abogado\Ky and Jil\Datasets\Cervical Cancer Dataset - Ranzeet Raut.csv')
    return df


# df = get_original_dataset()
df = pd.read_csv(r"C:\Cus\Jilliana Abogado\updated_imputed_missing_values_&_dropped_irrelevant_columns.csv")


# Check for missing values per column
missing_counts = df.isnull().sum()

# Display columns with missing values
print("Missing values per column:")
print(missing_counts)

# Optionally, show only columns that have at least one missing value
missing_columns = missing_counts[missing_counts > 0]
print("\nColumns that contain missing values:")
print(missing_columns)
