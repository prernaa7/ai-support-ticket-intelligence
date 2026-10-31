
import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "models/ticket_embeddings.index"
METADATA_PATH = "models/ticket_metadata.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Similarity threshold for suggesting a resolution
SIMILARITY_THRESHOLD = 0.30


def suggest_resolution(query):
    if not query.strip():
        print("Please enter a ticket description.")
        return

    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH, "rb") as file:
        metadata = pickle.load(file)

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, 1)

    score = float(scores[0][0])
    idx = int(indices[0][0])

    print("\nRESOLUTION SUGGESTION")
    print("=" * 50)

    if idx < 0 or score < SIMILARITY_THRESHOLD:
        print("No sufficiently similar historical ticket found.")
        return

    ticket = metadata[idx]

    print("New Ticket:", query.strip())
    print("\nClosest Historical Ticket:", ticket["ticket_id"])
    print("Similarity Score:", f"{score * 100:.2f}%")
    print("Category:", ticket["category"])
    print("Priority:", ticket["priority"])
    print("\nSuggested Resolution:")
    print(ticket["resolution"])

    print("\nNote: This is a recommendation based on a similar past ticket.")
    print("Verify the resolution before applying it.")


if __name__ == "__main__":
    new_ticket = """
    My payment was deducted from my bank account,
    but the transaction failed and my order was not placed.
    Please help me get my money refunded.
    """

    suggest_resolution(new_ticket)
