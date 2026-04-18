from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import json
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

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