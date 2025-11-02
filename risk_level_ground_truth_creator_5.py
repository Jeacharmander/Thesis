import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# 1️⃣ Load your real dataset
df = pd.read_csv(r"C:\Users\jycab\Documents\Thesis\dropped_columns_according_to_doc.csv")

# 2️⃣ Define metadata (so SDV knows column types)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# 3️⃣ Create and train synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(df)

# 4️⃣ Generate synthetic samples
synthetic_df = synthesizer.sample(num_rows=len(df))

# 5️⃣ Save
synthetic_df.to_csv(r"C:\Users\jycab\Documents\Thesis\synthetic_cervical_data.csv", index=False)
print("✅ Synthetic data saved as synthetic_cervical_data.csv")

print("\nOriginal class distribution:")
print(df["risk_level"].value_counts())
print("\nSynthetic class distribution:")
print(synthetic_df["risk_level"].value_counts())
