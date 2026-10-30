
import joblib

from app.preprocess import clean_text


# Load trained model
model = joblib.load("models/category_classifier.pkl")


# New support ticket
ticket = "I cannot login to my account. Password is not working."


# Clean ticket
cleaned_ticket = clean_text(ticket)


# Predict category
prediction = model.predict([cleaned_ticket])


# Display result
print("\nOriginal Ticket:", ticket)
print("Cleaned Ticket:", cleaned_ticket)
print("Predicted Category:", prediction[0])
