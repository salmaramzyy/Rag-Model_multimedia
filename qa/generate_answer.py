import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import os

# Load model 
model = SentenceTransformer('all-MiniLM-L6-v2')

client = OpenAI()

EMBEDDINGS_PATH = "outputs/egypt_embeddings.json"


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, top_k=5):
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    query_embedding = model.encode(query)

    results = []

    for item in data:
        score = cosine_similarity(query_embedding, np.array(item["embedding"]))

        results.append({
            "score": score,
            "content": item["content"],
            "page": item["page"]
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


generator = pipeline("text-generation", model="distilgpt2")

def generate_answer(query, contexts):
    context_text = "\n\n".join(
        [f"(Page {c['page']}) {c['content']}" for c in contexts]
    )

    prompt = f"""
Answer the question based on the context below.

Context:
{context_text}

Question:
{query}

Answer:
"""

    response = generator(prompt, max_length=300, do_sample=False)

    return response[0]["generated_text"]


if __name__ == "__main__":
    query = input("Enter your question: ")

    retrieved_chunks = retrieve(query)

    answer = generate_answer(query, retrieved_chunks)

    print("\n Answer:\n")
    print(answer)

    print("\n Sources:")
    for c in retrieved_chunks:
        print(f"Page {c['page']}")

    #  CREATE OUTPUT FOLDER
    os.makedirs("outputs", exist_ok=True)

    #  SAFE FILE NAME
    filename = (
        query.replace(" ", "_")
        .replace("?", "")
        .replace("/", "")
        .replace("\\", "")
        .replace(":", "")
    )

    #  STRUCTURED DATA (IMPORTANT)
    output_data = {
        "query": query,
        "answer": answer,
        "sources": [c["page"] for c in retrieved_chunks]
    }

    #  SAVE JSON
    with open(f"outputs/qa_{filename}.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    #  SAVE TXT (READABLE)
    with open(f"outputs/qa_{filename}.txt", "w", encoding="utf-8") as f:
        f.write(f"Question: {query}\n\n")
        f.write("Answer:\n")
        f.write(answer + "\n\n")
        f.write("Sources:\n")
        for c in retrieved_chunks:
            f.write(f"Page {c['page']}\n")

    print(f"\n QA saved as outputs/qa_{filename}.json & .txt")