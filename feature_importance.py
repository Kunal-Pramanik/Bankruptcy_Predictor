import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data.csv")

X = df.drop("Bankrupt?", axis=1)
y = df["Bankrupt?"]

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(20))

# visualization
top_features = feature_importance.head(20)

plt.figure(figsize=(10,8))
sns.barplot(x="Importance", y="Feature", data=top_features)

plt.title("Top 20 Important Features")
plt.show()