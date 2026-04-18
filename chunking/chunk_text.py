import json
import os

INPUT_PATH = "outputs/egypt_report.json"
OUTPUT_PATH = "outputs/egypt_chunks.json"

CHUNK_SIZE = 300  # words per chunk


def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def create_chunks():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []

    for page in pages:
        page_num = page["page"]
        text = page["text"]

        text_chunks = chunk_text(text, CHUNK_SIZE)

        for chunk in text_chunks:
            chunk_data = {
                "page": page_num,
                "content": chunk
            }
            all_chunks.append(chunk_data)

    return all_chunks


if __name__ == "__main__":
    chunks = create_chunks()

    os.makedirs("outputs", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f" Created {len(chunks)} chunks")