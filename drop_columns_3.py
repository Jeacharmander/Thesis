import pandas as pd

df = pd.read_csv(
    r"C:\Cus\Jilliana Abogado\original_dataset_with_empty_values_handled.csv"
)

for i in df.columns:
    print(i)

df.drop(
    columns=[
            "IUD",
            "STDs:condylomatosis",
            "STDs:molluscum contagiosum",
            "STDs:AIDS",
            "Dx:Cancer",
            "Dx:CIN",
            "Dx:HPV",
            "Dx",
            "Hinselmann",
            "Schiller",
            "Citology",
            "Biopsy"
    ],
    inplace=True,
)

for i in df.columns:
    print(i)
# df.to_csv(r"C:\Cus\Jilliana Abogado\dropped_columns_according_to_doc.csv", index=False)
print("Columns dropped and was saved as dropped_columns_according_to_doc.csv")

