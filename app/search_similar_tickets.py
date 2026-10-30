
import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "models/ticket_embeddings.index"
METADATA_PATH = "models/ticket_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def search_similar_tickets(query, top_k=3):

    print("\nLoading FAISS index...")

    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH, "rb") as file:
        metadata = pickle.load(file)

    print("Loading Sentence Transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    print("\nConverting new ticket into embedding...")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    print("Searching for similar tickets...")

    scores, indices = index.search(query_embedding, top_k)

    print("\nSIMILAR SUPPORT TICKETS")
    print("=" * 60)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        ticket = metadata[idx]

        print(f"\nMatch {rank}")
        print("-" * 40)
        print("Ticket ID:", ticket["ticket_id"])
        print("Similarity Score:", f"{score * 100:.2f}%")
        print("Description:", ticket["description"])
        print("Category:", ticket["category"])
        print("Priority:", ticket["priority"])
        print("Resolution:", ticket["resolution"])


if __name__ == "__main__":

    new_ticket = """
    My payment was deducted from my bank account,
    but the transaction failed and my order was not placed.
    Please help me get my money refunded.
    """

    print("NEW CUSTOMER SUPPORT TICKET")
    print("=" * 60)
    print(new_ticket)

    search_similar_tickets(new_ticket)
