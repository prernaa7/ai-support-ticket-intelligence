
import re


def clean_text(text):

    # Convert text to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Test preprocessing
sample = "LOGIN ERROR!!! Password Reset 123"

print("Original:", sample)
print("Cleaned:", clean_text(sample))
