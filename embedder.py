from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_into_chunks(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings

def save_index(chunks, embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "faiss.index")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    text = extract_text_from_pdf("data/IS456.pdf")
    chunks = split_into_chunks(text)
    embeddings = embed_chunks(chunks)
    save_index(chunks, embeddings)
    print("✅ PDF processed and stored!")
