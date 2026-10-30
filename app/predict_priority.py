
import joblib

from app.preprocess import clean_text


# 1. Load saved model components
saved_data = joblib.load("models/priority_classifier.pkl")

model = saved_data["model"]
vectorizer = saved_data["vectorizer"]
label_encoder = saved_data["label_encoder"]


# 2. New support ticket
ticket = (
    "My bank payment was deducted but the transaction "
    "failed and my money has not been refunded."
)


# 3. Clean ticket
cleaned_ticket = clean_text(ticket)


# 4. Convert text into TF-IDF features
ticket_features = vectorizer.transform([cleaned_ticket])


# 5. Predict priority
prediction = model.predict(ticket_features)

priority = label_encoder.inverse_transform(
    prediction.astype(int)
)


# 6. Display output
print("\nOriginal Ticket:", ticket)
print("Cleaned Ticket:", cleaned_ticket)
print("Predicted Priority:", priority[0])
