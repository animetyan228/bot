from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz
import faiss
import numpy as np
from llama_cpp import Llama
from crewai import Agent, Task, Crew
import os
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# --- Путь к локальной модели Mistral 7B Instruct (GGUF) ---
MODEL_PATH = r"./model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=60,
    n_threads=12,
    n_ctx=4096,
    use_mlock=True,
    use_mmap=True,
    verbose=False
)


# --- PDF чтение ---
def read_pdf(pdf_path):
    doc = fitz.Document(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

# --- Разбиение текста ---
def split_text(text, chunk_size=600):
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        chunk.append(word)
        if len(chunk) >= chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# --- Векторизация (SentenceTransformer) ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text_list):
    vectors = embed_model.encode(text_list)
    return np.array(vectors).astype("float32")

# --- Создание FAISS ---
def create_vectorstore(chunks):
    vectors = embed(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

# --- RAG поиск ---
def rag_search(query, chunks, index, vectors, top_k=3):
    q_vector = embed([query])
    distances, ids = index.search(q_vector, top_k)
    results = [chunks[int(i)] for i in ids[0]]
    return "\n\n---\n".join(results)

# --- Agent (CrewAI) ---
rag_agent = Agent(
    role="RAG Document Analyst",
    goal="Использовать только предоставленный контекст для точного ответа.",
    verbose=False,
    backstory="Эксперт по анализу PDF.",
    allow_delegation=False
)

# --- Генерация ответа через локальную Mistral ---
def generate_text(prompt, max_tokens=512):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["</s>", "###"],
        echo=False
    )
    return output["choices"][0]["text"]

# --- Flask endpoint ---
@app.route("/", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Укажи message"}), 400
    if not data.get("pdf_path"):
        return jsonify({"error": "Укажи pdf_path"}), 400

    question = data["message"]
    pdf_path = data["pdf_path"]

    # Читаем PDF
    try:
        pdf_text = read_pdf(pdf_path)
    except Exception as e:
        return jsonify({"error": f"Ошибка чтения PDF: {e}"}), 500

    # RAG подготовка
    chunks = split_text(pdf_text)
    index, vectors = create_vectorstore(chunks)
    context = rag_search(question, chunks, index, vectors)

    final_prompt = (
        "Ты — эксперт по анализу документов.\n"
        "Используй строго только информацию из контекста.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        "Ответ:"
    )

    # Генерация
    try:
        answer = generate_text(final_prompt)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
