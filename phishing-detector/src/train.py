import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from extract_features import extract_url_features

# 1) Load dataset (tiny demo set; extend it for better accuracy)
df = pd.read_csv("data/sample_urls.csv")

# 2) Extract features
X = pd.DataFrame([extract_url_features(u) for u in df["url"]])
y = df["label"].astype(int)

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4) Train model
model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

# 5) Evaluate
y_pred = model.predict(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# 6) Save model
joblib.dump(model, "model.pkl")
print("\n✅ Saved model as model.pkl")
