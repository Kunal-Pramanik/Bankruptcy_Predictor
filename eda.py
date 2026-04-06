import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data.csv")

# target distribution
print(df["Bankrupt?"].value_counts())

print("\nPercentage:")
print(df["Bankrupt?"].value_counts(normalize=True)*100)

# plot class distribution
sns.countplot(x="Bankrupt?", data=df)
plt.title("Bankruptcy Distribution")
plt.show()

# correlation with target
corr_target = df.corr()["Bankrupt?"].sort_values(ascending=False)

print("\nTop positive correlations:")
print(corr_target.head(10))

print("\nTop negative correlations:")
print(corr_target.tail(10))