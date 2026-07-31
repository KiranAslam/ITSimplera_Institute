from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR_CANDIDATES = [ROOT_DIR / "data", ROOT_DIR / "Data"]
DATA_PATH = ROOT_DIR / "data" / "amazon_reviews.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.joblib"
POS_WORDCLOUD_PATH = ROOT_DIR / "models" / "positive_reviews_wordcloud.png"
NEG_WORDCLOUD_PATH = ROOT_DIR / "models" / "negative_reviews_wordcloud.png"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for", "from",
    "had", "has", "have", "i", "in", "is", "it", "its", "just", "me", "my", "not", "of", "on",
    "or", "our", "so", "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your", "very", "really", "can",
    "could", "love", "loved", "good", "great"
}


def resolve_data_path() -> Path:
    for candidate in DATA_DIR_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            csv_file = candidate / "amazon_reviews.csv"
            if csv_file.exists():
                return csv_file
    return DATA_PATH


def ensure_dataset_copy() -> Path:
    source_path = resolve_data_path()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if source_path != DATA_PATH and not DATA_PATH.exists():
        shutil.copy2(source_path, DATA_PATH)
    return DATA_PATH


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOPWORDS and len(token) > 1]
    return " ".join(tokens)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else ensure_dataset_copy()
    df = pd.read_csv(data_path)
    review_col = "verified_reviews" if "verified_reviews" in df.columns else next((c for c in df.columns if "review" in c.lower()), None)
    label_col = "feedback" if "feedback" in df.columns else next((c for c in df.columns if "label" in c.lower() or "sentiment" in c.lower()), None)

    if review_col is None or label_col is None:
        raise ValueError(f"Expected review and label columns in dataset. Found: {df.columns.tolist()}")

    df = df[[review_col, label_col]].copy()
    df.columns = ["verified_reviews", "feedback"]
    df = df.dropna(subset=["verified_reviews", "feedback"])
    df["cleaned_review"] = df["verified_reviews"].apply(clean_text)
    df["sentiment"] = df["feedback"].map({1: "positive", 0: "negative"})
    df = df[df["sentiment"].notna()].reset_index(drop=True)
    return df


def create_wordcloud(texts: pd.Series, output_path: Path, title: str) -> None:
    text = " ".join(texts.astype(str))
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_models(df: pd.DataFrame) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_review"], df["feedback"], test_size=0.2, random_state=42, stratify=df["feedback"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=3000, random_state=42),
        "multinomial_nb": MultinomialNB(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        results[name] = {
            "model": model,
            "predictions": preds,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "report": classification_report(y_test, preds, target_names=["negative", "positive"], output_dict=True),
            "cm": confusion_matrix(y_test, preds),
        }

    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = results[best_name]["model"]
    best_result = results[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(best_model, MODEL_PATH)

    positive_reviews = df.loc[df["feedback"] == 1, "cleaned_review"]
    negative_reviews = df.loc[df["feedback"] == 0, "cleaned_review"]
    create_wordcloud(positive_reviews, POS_WORDCLOUD_PATH, "Positive Reviews Word Cloud")
    create_wordcloud(negative_reviews, NEG_WORDCLOUD_PATH, "Negative Reviews Word Cloud")

    return {
        "vectorizer": vectorizer,
        "results": results,
        "best_name": best_name,
        "best_result": best_result,
        "X_test": X_test,
        "y_test": y_test,
    }


def plot_confusion_matrix(cm: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()


def predict_sentiment(text: str) -> Tuple[str, float]:
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    probability = model.predict_proba(features)[0]
    predicted_index = int(model.predict(features)[0])
    sentiment = "positive" if predicted_index == 1 else "negative"
    confidence = float(probability.max())
    return sentiment, confidence
