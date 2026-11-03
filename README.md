# AI Support Ticket Intelligence System

An AI-powered application that classifies support tickets, predicts priority, summarizes issues, suggests resolutions, and retrieves similar tickets and knowledge-base articles.

## Features

- Ticket category classification using TF-IDF and Logistic Regression
- Priority prediction using XGBoost
- AI-powered summarization using Gemini
- Similar-ticket retrieval and resolution suggestions using FAISS
- Semantic knowledge-base search
- Flask REST API and Streamlit UI
- Dockerized deployment using Docker Compose

## Tech Stack

Python, Pandas, Scikit-learn, XGBoost, Gemini API, Sentence Transformers, FAISS, Flask, Streamlit, Docker.

## Dataset

- 12 sample tickets across 4 categories: Login, Payment, Technical, and Account
- Includes ticket descriptions, priorities, and resolutions

*The dataset is a small sample for development and testing.*

## Setup

1. Install Docker Desktop.
2. Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. Build and start the application:

   ```bash
   docker compose up --build
   ```

4. Open:
   - UI: http://localhost:8501
   - API: http://localhost:5000
   - Health check: http://localhost:5000/health

Stop the containers with:

```bash
docker compose down
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict/category` | Predict category |
| POST | `/predict/priority` | Predict priority |
| POST | `/summarize` | Summarize ticket |
| POST | `/similar` | Find similar tickets |
| POST | `/resolution` | Suggest resolution |
| POST | `/knowledge-base` | Search knowledge base |

## Testing

Tested API endpoints, Streamlit UI, empty-input validation, Login and Payment tickets, and Docker Compose startup.

## Future Improvements

- Expand the dataset
- Improve model evaluation
- Add authentication and logging
- Deploy to the cloud
