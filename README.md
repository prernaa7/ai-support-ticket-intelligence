# AI Support Ticket Intelligence System

An AI-powered project to classify support tickets, predict priority, summarize issues, suggest resolutions, and retrieve similar historical tickets.

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
- Sentence Transformers
- Flask

## Features

### Ticket Category Classification

- Implemented text preprocessing using Python and Regex.
- Converted ticket descriptions into numerical features using TF-IDF.
- Built a Logistic Regression classification model to predict support ticket categories.
- Evaluated model performance using accuracy and a classification report.
- Saved the trained ML pipeline using Joblib.
- Tested predictions on new support ticket descriptions.

**Technologies:** Python, Pandas, Scikit-learn, TF-IDF, Logistic Regression, Joblib.

### Ticket Priority Prediction

- Inspected support ticket priority distribution.
- Encoded priority labels using LabelEncoder.
- Converted ticket descriptions into numerical features using TF-IDF.
- Built an XGBoost classification model to predict High, Medium, and Low priorities.
- Evaluated model performance using accuracy and a classification report.
- Saved the trained model using Joblib.
- Tested priority predictions on new support tickets.

**Technologies:** Python, Pandas, Scikit-learn, XGBoost, TF-IDF, LabelEncoder, Joblib.

### AI-Powered Ticket Summarization

- Integrated Google's Gemini API using the Google GenAI Python SDK.
- Configured secure API key management using environment variables and Python Dotenv.
- Created a ticket summarization function.
- Used Gemini Flash-Lite to generate concise summaries of customer support tickets.
- Extracted the main issue, important details, and customer requirements.
- Tested summarization using a sample payment failure ticket.

**Technologies:** Python, Gemini API, Google GenAI SDK, Python Dotenv, Generative AI.

### Similar Ticket Retrieval

- Generated numerical embeddings from support ticket descriptions using Sentence Transformers.
- Built a FAISS vector index to store and search ticket embeddings.
- Implemented similarity search to identify previously reported customer issues.
- Retrieved the top 3 most similar historical support tickets.
- Displayed matching ticket IDs, descriptions, categories, and resolutions.
- Tested retrieval using a new payment failure support ticket.

**Technologies:** Python, FAISS, Sentence Transformers, Vector Embeddings, Semantic Search, Pandas.

### Resolution Suggestion

- Reused the FAISS vector index and Sentence Transformers to find the closest historical support ticket.
- Suggested a resolution based on the most similar ticket when its similarity score exceeded a threshold.
- Displayed the matching ticket ID, similarity score, category, and priority.
- Handled empty input and cases where no sufficiently similar ticket was found.
- Tested with a payment failure ticket and retrieved T005 with the suggested resolution: “Verify transaction and process refund if needed.”

**Technologies:** Python, FAISS, Sentence Transformers, Semantic Search.

## Planned Features

- Knowledge-base retrieval
- Flask backend API
- Streamlit user interface
- Testing and debugging
- Docker deployment
