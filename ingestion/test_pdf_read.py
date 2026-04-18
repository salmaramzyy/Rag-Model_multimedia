import fitz  # PyMuPDF
import os
import json

PDF_PATH = "data/pdfs/egypt_report.pdf"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    
    all_pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        page_data = {
            "page": page_num + 1,
            "text": text
        }

        all_pages.append(page_data)

    return all_pages


if __name__ == "__main__":
    pages = extract_text_from_pdf(PDF_PATH)

    # 1. Print preview (for debugging)
    for page in pages[:2]:
        print(f"\n--- Page {page['page']} ---")
        print(page["text"][:500])

    # 2. Save as JSON (IMPORTANT for next steps)
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/egypt_report.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    # 3. Save as TXT
    with open("outputs/egypt_report.txt", "w", encoding="utf-8") as f:
        for page in pages:
            f.write(f"\n--- Page {page['page']} ---\n")
            f.write(page["text"])

    print("\n Extraction completed and saved in 'outputs/' folder")