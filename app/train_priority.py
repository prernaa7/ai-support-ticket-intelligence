
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

from app.preprocess import clean_text


# 1. Load dataset
df = pd.read_csv("data/tickets.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# 2. Check missing values
df = df.dropna(subset=["description", "priority"])

# 3. Clean descriptions
df["description"] = df["description"].apply(clean_text)

# 4. Separate input and target
X = df["description"]
y = df["priority"].str.strip().str.title()

print("\nPriority Distribution:")
print(y.value_counts())


# 5. Encode priority labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nPriority Mapping:")
for number, label in enumerate(label_encoder.classes_):
    print(number, "=", label)


# 6. Check whether stratified splitting is possible
if y.value_counts().min() < 2:
    raise ValueError(
        "At least 2 records per priority class are required. "
        "Add more examples to tickets.csv before training."
    )

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 8. Convert text into numerical features
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 9. Create XGBoost model
model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


# 10. Train model
model.fit(X_train_tfidf, y_train)

print("\nXGBoost training completed!")


# 11. Make predictions
predictions = model.predict(X_test_tfidf)


# 12. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nPriority Model Accuracy:",
      round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))


# 13. Save complete model components
joblib.dump(
    {
        "model": model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder
    },
    "models/priority_classifier.pkl"
)

print("\nPriority model saved successfully!")
