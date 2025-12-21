# инструменты для crew ai

from crewai.tools import tool
from core.pdf_utils import read_pdf
from core.rag_utils import make_chunks, build_faiss_index, rag_search
from core.vector_store import save_index, get_index


# инструмент для чтения пдф
@tool("PDF Reader Tool")
def pdf_reader_tool(pdf_path: str) -> str:
    """читает pdf и возвращает текст, плюс печатает диагностику"""
    text = read_pdf(pdf_path)
    return text



# инструмент для чанков
@tool("Chunking Tool")
def chunk_tool(text: str, chunk_size: int = 600) -> str:
    """делит текст на чанки и возвращает их одной строкой."""
    chunks = make_chunks(text, chunk_size=chunk_size)
    return "\n\n[CHUNK_SEPARATOR]\n\n".join(chunks)


# инструмент векторизации
@tool("Vector Store Builder Tool")
def build_vector_index_tool(text: str) -> str:
    """строит faiss индекс по тексту и возвращает index_id."""
    chunks = make_chunks(text)
    index, vectors = build_faiss_index(chunks)
    return save_index(chunks, index, vectors)


# инструмент rag
@tool("RAG Search Tool")
def rag_search_tool(index_id: str, query: str, top_k: int = 3) -> str:
    """ищет top_k фрагментов по запросу и возвращает контекст."""
    stored = get_index(index_id)
    if stored is None:
        return ""
    chunks, index, vectors = stored
    return rag_search(chunks, index, vectors, query, top_k=top_k)
