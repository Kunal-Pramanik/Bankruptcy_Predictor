import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("cleaned_data.csv")

X = df.drop("Bankrupt?", axis=1)
y = df["Bankrupt?"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

top_features = importance.head(15)

print(top_features)

top_features["Feature"].to_csv("top_features.csv", index=False)