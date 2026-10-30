
import os
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/tickets.csv"
INDEX_PATH = "models/ticket_embeddings.index"
METADATA_PATH = "models/ticket_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index():
    print("Loading ticket dataset...")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["description"]).reset_index(drop=True)

    print("Total tickets:", len(df))

    print("\nLoading Sentence Transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    descriptions = df["description"].astype(str).tolist()

    print("\nGenerating ticket embeddings...")
    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    print("Embedding shape:", embeddings.shape)

    print("\nBuilding FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs("models", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    metadata = df[
        ["ticket_id", "description", "category", "priority", "resolution"]
    ].to_dict(orient="records")

    with open(METADATA_PATH, "wb") as file:
        pickle.dump(metadata, file)

    print("\nFAISS index created successfully!")
    print("Total indexed tickets:", index.ntotal)
    print("Index saved to:", INDEX_PATH)
    print("Metadata saved to:", METADATA_PATH)


if __name__ == "__main__":
    build_index()
