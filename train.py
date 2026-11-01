
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
data_path = Path(__file__).parent / "data" / "SMSSpamCollection"

df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

# Remove missing and duplicate records
df = df.dropna().drop_duplicates()

print("Dataset size:", len(df))
print(df["label"].value_counts())

# Separate features and target
X = df["message"]
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Text vectorization and classification
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, y_pred,
    labels=["ham", "spam"]
))

# Save trained model
joblib.dump(
    model,
    Path(__file__).parent / "spam_model.joblib"
)

print("\nModel saved successfully!")
