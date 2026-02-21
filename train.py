import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

print("Loading training data...")
df = pd.read_csv("data/crop_recommendation.csv")

X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

os.makedirs("models", exist_ok=True)

# Save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

models = {
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":            XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss"),
    "SVMClassifier":      SVC(kernel="rbf", random_state=42),
    "DecisionTree":       DecisionTreeClassifier(random_state=42),
    "NBClassifier":       GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  {name} accuracy: {acc:.4f}")
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

print("All models trained and saved.")
