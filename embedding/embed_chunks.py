import json
import os
from sentence_transformers import SentenceTransformer

INPUT_PATH = "outputs/egypt_chunks.json"
OUTPUT_PATH = "outputs/egypt_embeddings.json"

# Load model 
model = SentenceTransformer('all-MiniLM-L6-v2')


def create_embeddings():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedded_data = []

    for i, chunk in enumerate(chunks):
        text = chunk["content"]

        embedding = model.encode(text).tolist()

        data = {
            "page": chunk["page"],
            "content": text,
            "embedding": embedding
        }

        embedded_data.append(data)

        if i % 50 == 0:
            print(f"Processed {i} chunks...")

    return embedded_data


if __name__ == "__main__":
    embeddings = create_embeddings()

    os.makedirs("outputs", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print(f" Saved {len(embeddings)} embeddings")