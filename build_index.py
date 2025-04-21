import faiss
import cohere
import os
import pickle
from dotenv import load_dotenv
import pdfplumber
import numpy as np
import time

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# Load PDF and extract text lines
def load_docs():
    all_text = ""
    with pdfplumber.open(r"C:\Users\A Indhumathi\OneDrive\Desktop\Projects\BUILDBOT\buildbot\data\is.456.2000.pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text.split("\n")

# Batch embed texts
def embed_texts(texts, batch_size=32):
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"\n🔄 Total chunks: {len(texts)} | Batch size: {batch_size} | Total batches: {total_batches}")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            print(f"⚙️ Embedding batch {i // batch_size + 1}/{total_batches}...")
            response = co.embed(texts=batch, model="embed-english-v3.0", input_type="search_document")
            all_embeddings.extend(response.embeddings)
            time.sleep(1)  # Pause to avoid hitting rate limits
        except Exception as e:
            print(f"❌ Error in batch {i // batch_size + 1}: {e}")
            continue

    return all_embeddings

# Build FAISS index
def build_faiss_index(text_chunks):
    embeddings = embed_texts(text_chunks)
    if not embeddings:
        print("❌ Error: No embeddings returned. Exiting.")
        return

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, "faiss.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(text_chunks, f)

    print("\n✅ FAISS index and chunks saved successfully!")

# Main
if __name__ == "__main__":
    chunks = load_docs()
    build_faiss_index(chunks)
