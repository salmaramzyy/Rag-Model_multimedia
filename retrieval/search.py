import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = "outputs/egypt_embeddings.json"

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, top_k=5):
    # Load embeddings
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Embed query
    query_embedding = model.encode(query)

    results = []

    for item in data:
        chunk_embedding = np.array(item["embedding"])
        score = cosine_similarity(query_embedding, chunk_embedding)

        results.append({
            "score": float(score),  
            "content": item["content"],
            "page": item["page"]
        })

    # Sort by similarity
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    query = input("Enter your question: ")

    results = search(query)

    print("\n Top Results:\n")

    for i, res in enumerate(results):
        print(f"Result {i+1} (Page {res['page']}, Score {res['score']:.4f})")
        print(res["content"][:300])
        print("-" * 50)

    # Create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # Save data
    output_data = {
        "query": query,
        "results": results
    }

    filename = (
        query.replace(" ", "_")
        .replace("?", "")
        .replace("/", "")
        .replace("\\", "")
        .replace(":", "")
    )

    # Save JSON
    with open(f"outputs/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Save TXT
    with open(f"outputs/{filename}.txt", "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        for i, res in enumerate(results):
            f.write(f"Result {i+1} (Page {res['page']}, Score {res['score']:.4f})\n")
            f.write(res["content"])
            f.write("\n" + "-" * 50 + "\n")

    print(f"\n Results saved as outputs/{filename}.json & .txt")