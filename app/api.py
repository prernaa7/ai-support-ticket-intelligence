
import os
import pickle

import faiss
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

from app.preprocess import clean_text
from app.summarize_ticket import summarize_ticket

app = Flask(__name__)

# Load trained models
category_model = joblib.load("models/category_classifier.pkl")

priority_data = joblib.load("models/priority_classifier.pkl")
priority_model = priority_data["model"]
priority_vectorizer = priority_data["vectorizer"]
priority_encoder = priority_data["label_encoder"]

# Load historical ticket index and metadata
ticket_index = faiss.read_index("models/ticket_embeddings.index")

with open("models/ticket_metadata.pkl", "rb") as file:
    ticket_metadata = pickle.load(file)

# Load embedding model once
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load knowledge base and create its vector index
knowledge_base = pd.read_csv("data/knowledge_base.csv")

article_texts = (
    knowledge_base["title"] + ". " + knowledge_base["content"]
).tolist()

article_embeddings = embedding_model.encode(
    article_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

knowledge_index = faiss.IndexFlatIP(article_embeddings.shape[1])
knowledge_index.add(article_embeddings)


def get_ticket():
    data = request.get_json(silent=True) or {}
    ticket = data.get("description", "").strip()

    if not ticket:
        return None, (
            jsonify({"error": "Please provide a ticket description."}),
            400
        )

    return ticket, None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict/category", methods=["POST"])
def predict_category():
    ticket, error = get_ticket()
    if error:
        return error

    prediction = category_model.predict([clean_text(ticket)])

    return jsonify({
        "ticket": ticket,
        "category": str(prediction[0])
    })


@app.route("/predict/priority", methods=["POST"])
def predict_priority():
    ticket, error = get_ticket()
    if error:
        return error

    cleaned_ticket = clean_text(ticket)
    features = priority_vectorizer.transform([cleaned_ticket])
    prediction = priority_model.predict(features)
    priority = priority_encoder.inverse_transform(prediction.astype(int))

    return jsonify({
        "ticket": ticket,
        "priority": str(priority[0])
    })


@app.route("/summarize", methods=["POST"])
def summarize():
    ticket, error = get_ticket()
    if error:
        return error

    summary = summarize_ticket(ticket)

    return jsonify({
        "ticket": ticket,
        "summary": summary
    })


@app.route("/similar", methods=["POST"])
def similar_tickets():
    ticket, error = get_ticket()
    if error:
        return error

    top_k = min(int((request.get_json(silent=True) or {}).get("top_k", 3)), 12)

    query_embedding = embedding_model.encode(
        [ticket],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = ticket_index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = ticket_metadata[int(idx)]
        results.append({
            "ticket_id": item["ticket_id"],
            "similarity": round(float(score), 4),
            "description": item["description"],
            "category": item["category"],
            "priority": item["priority"],
            "resolution": item["resolution"]
        })

    return jsonify({"results": results})


@app.route("/resolution", methods=["POST"])
def resolution():
    ticket, error = get_ticket()
    if error:
        return error

    query_embedding = embedding_model.encode(
        [ticket],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = ticket_index.search(query_embedding, 1)

    score = float(scores[0][0])
    idx = int(indices[0][0])

    if idx < 0 or score < 0.30:
        return jsonify({
            "ticket": ticket,
            "message": "No sufficiently similar historical ticket found."
        })

    item = ticket_metadata[idx]

    return jsonify({
        "ticket": ticket,
        "matched_ticket_id": item["ticket_id"],
        "similarity": round(score, 4),
        "category": item["category"],
        "priority": item["priority"],
        "suggested_resolution": item["resolution"],
        "note": "Verify the suggested resolution before applying it."
    })


@app.route("/knowledge-base", methods=["POST"])
def knowledge_base_search():
    ticket, error = get_ticket()
    if error:
        return error

    query_embedding = embedding_model.encode(
        [ticket],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    top_k = min(int((request.get_json(silent=True) or {}).get("top_k", 3)),
                 len(knowledge_base))

    scores, indices = knowledge_index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        article = knowledge_base.iloc[int(idx)]
        results.append({
            "article_id": article["article_id"],
            "title": article["title"],
            "similarity": round(float(score), 4),
            "category": article["category"],
            "content": article["content"]
        })

    return jsonify({"articles": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
