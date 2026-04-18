import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
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


# UI
st.title(" Multi-Modal RAG QA System")

query = st.text_input("Ask a question about the document:")

if query:
    results = retrieve(query)

    st.subheader(" Retrieved Context")

    for res in results:
        st.write(f"**Page {res['page']} (Score: {res['score']:.3f})**")
        st.write(res["content"][:300])
        st.write("---")

    answer = ""
    for r in results[:2]:
        answer += f"(Page {r['page']}) {r['content']}\n\n"

    st.subheader(" Generated Answer")
    st.write(answer)

    st.subheader(" Sources")
    for r in results:
        st.write(f"Page {r['page']}")