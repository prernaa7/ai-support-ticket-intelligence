# AI Support Ticket Intelligence System

An AI-powered project to classify support tickets, predict priority, summarize issues, and retrieve similar historical tickets.

## Day 1: Project Setup and Dataset Loading

- Created the project structure and Python virtual environment.
- Installed Pandas and NumPy.
- Loaded a CSV dataset using Pandas.
- Inspected dataset shape, column types, ticket categories, priorities, and missing values.

## Dataset

- 12 sample support tickets
- 5 columns: ticket ID, description, category, priority, and resolution
- 4 categories: Login, Payment, Technical, Account
- 3 priority levels: High, Medium, Low
- Missing values: 0

**Note:** The current dataset is a small sample for development and testing.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Gemini API
- FAISS
- Flask

## Planned Features

1. Ticket category classification
2. Ticket priority prediction
3. AI-powered ticket summarization
4. Similar-ticket retrieval
5. API deployment



### Day 2: Ticket Preprocessing and Classification

- Implemented text preprocessing using Python and Regex.
- Converted ticket descriptions into numerical features using TF-IDF.
- Built a Logistic Regression classification model.
- Trained the model to predict support ticket categories.
- Evaluated model performance using accuracy and classification report.
- Saved the trained ML pipeline using Joblib.
- Tested predictions on new support ticket descriptions.

**Technologies:** Python, Pandas, Scikit-learn, TF-IDF,
Logistic Regression, Joblib.


### Day 3: Priority Prediction using XGBoost

- Inspected support ticket priority distribution.
- Preprocessed ticket descriptions.
- Encoded priority labels using LabelEncoder.
- Converted ticket descriptions into numerical features using TF-IDF.
- Built an XGBoost classification model.
- Trained the model to predict High, Medium and Low priorities.
- Evaluated model performance using accuracy and classification report.
- Saved the trained model using Joblib.
- Tested priority predictions on new support tickets.

**Technologies:** Python, Pandas, Scikit-learn,
XGBoost, TF-IDF, LabelEncoder, Joblib.
