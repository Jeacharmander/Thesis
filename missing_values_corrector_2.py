""" 
    This program handles all the missing values (Na or NaN) in the original dataset.
"""

from columns_checker_1 import get_original_dataset

# Load your CSV file
df = get_original_dataset()

# Replace all NaN or empty values with 0
df_filled = df.fillna(0)

# Save back to a new CSV file (optional)
df_filled.to_csv(r'C:\Cus\Jilliana Abogado\original_dataset_with_empty_values_handled.csv', index=False)

print("All missing values have been replaced with 0.")
