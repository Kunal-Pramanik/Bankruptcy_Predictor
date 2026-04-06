import pandas as pd

df = pd.read_csv("taiwanese+bankruptcy+prediction.csv")

# remove extra spaces in column names
df.columns = df.columns.str.strip()

# check missing values
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])

# check duplicates
duplicates = df.duplicated().sum()
print("\nDuplicate rows:", duplicates)

# drop duplicates if any
df = df.drop_duplicates()

print("\nCleaned dataset shape:", df.shape)


df.to_csv("cleaned_data.csv", index=False)