import cohere
import faiss
import pickle
from dotenv import load_dotenv
import os
import numpy as np

# Load .env and get API key
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

def load_faiss_index():
    index = faiss.read_index("faiss.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def embed_text(texts):
    """
    Generate embeddings using Cohere.
    Input: List of strings
    Output: List of embedding vectors (NumPy array)
    """
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_query"  # ✅ Important!
    )
    # Convert list of embeddings to a NumPy array
    return np.array(response.embeddings)

def get_answer(question):
    question_embedding = embed_text([question])

    index, chunks = load_faiss_index()
    # Make sure question_embedding is in the correct shape for FAISS
    question_embedding = question_embedding.astype(np.float32)  # Ensure the type is float32 for FAISS
    D, I = index.search(question_embedding, k=3)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"Use the following construction code content to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    response = co.generate(
        model="command",  # Use your custom model name here
        prompt=prompt,
        max_tokens=300,
        temperature=0.3,
    )
    return response.generations[0].text.strip()

# Optional: for CLI testing
if __name__ == "__main__":
    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        try:
            answer = get_answer(query)
            print("\n💬 Answer:\n", answer)
        except Exception as e:
            print("⚠️ Error:", e)
