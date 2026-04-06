import pandas as pd

# load dataset
df = pd.read_csv("taiwanese+bankruptcy+prediction.csv")


print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe().T)