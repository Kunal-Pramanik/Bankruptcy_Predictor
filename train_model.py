import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("cleaned_data.csv")

top_features = pd.read_csv("top_features.csv")["Feature"].tolist()

X = df[top_features]
y = df["Bankrupt?"]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "bankruptcy_model.pkl")

print("Model saved successfully")