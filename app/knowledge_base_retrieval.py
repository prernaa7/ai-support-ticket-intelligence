import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/knowledge_base.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def retrieve_articles(query, top_k=3):
    if not query.strip():
        print("Please enter a ticket description.")
        return

    knowledge_base = pd.read_csv(DATA_PATH)

    model = SentenceTransformer(MODEL_NAME)

    article_texts = (
        knowledge_base["title"] + ". " + knowledge_base["content"]
    ).tolist()

    article_embeddings = model.encode(
        article_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(article_embeddings.shape[1])
    index.add(article_embeddings)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(knowledge_base))
    )

    print("\nRELEVANT KNOWLEDGE-BASE ARTICLES")
    print("=" * 50)

    for score, idx in zip(scores[0], indices[0]):
        article = knowledge_base.iloc[int(idx)]

        print(f"\nArticle: {article['article_id']} - {article['title']}")
        print(f"Similarity: {score * 100:.2f}%")
        print(f"Category: {article['category']}")
        print(f"Content: {article['content']}")


if __name__ == "__main__":
    sample_ticket = """
    My payment was deducted from my bank account,
    but the transaction failed and my order was not placed.
    """

    retrieve_articles(sample_ticket)