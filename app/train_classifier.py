
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from app.preprocess import clean_text


# 1. Load dataset
df = pd.read_csv("data/tickets.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# 2. Clean ticket descriptions
df["description"] = df["description"].apply(clean_text)


# 3. Separate input and output
X = df["description"]
y = df["category"]


# 4. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 5. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# 6. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")


# 7. Make predictions
predictions = model.predict(X_test)


# 8. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))


# 9. Save trained model
joblib.dump(model, "models/category_classifier.pkl")

print("\nModel saved successfully!")
