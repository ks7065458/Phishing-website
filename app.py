from flask import Flask, request, render_template, jsonify
import os
import joblib
import requests
from src.extract_features import extract_url_features

app = Flask(__name__)

MODEL_PATH = "model.pkl"

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

def is_reachable(url: str) -> bool:
    try:
        u = normalize_url(url)
        r = requests.head(u, timeout=5, allow_redirects=True)
        return r.status_code < 400
    except Exception:
        return False

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

def predict_url(url: str):
    # 1) Check reachability first
    reachable = is_reachable(url)
    if not reachable:
        return "Unreachable Website", "N/A", {"status": "unreachable"}

    # 2) Extract features and predict
    u = normalize_url(url)
    feats = extract_url_features(u)
    X = [list(feats.values())]

    if model:
        prob = float(model.predict_proba(X)[0][1])  # probability of "phishing" class
        pred = "Phishing" if prob > 0.5 else "Legitimate"
        return pred, round(prob, 2), feats

    # 3) Fallback heuristic if model missing
    score = 0
    if feats["has_suspicious_words"]: score += 2
    if feats["is_shortened"]: score += 1
    if feats["has_ip"]: score += 2
    if feats["num_subdomains"] >= 3: score += 1
    pred = "Phishing" if score >= 2 else "Legitimate"
    return pred, "N/A (heuristic)", feats

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        pred, prob, feats = predict_url(url)
        return render_template("index.html", url=url, pred=pred, prob=prob, feats=feats)
    return render_template("index.html")

@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    pred, prob, feats = predict_url(url)
    return jsonify({"url": url, "prediction": pred, "probability": prob, "features": feats})

@app.route("/healthz")
def health():
    return {"ok": True}, 200

if __name__ == "__main__":
    # For local dev only. On Heroku we use gunicorn via Procfile.
    app.run(host="0.0.0.0", port=8080, debug=True)
