import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "space_knowledge.txt")


def build_vector_store():
    """Load documents, split, embed, and persist to ChromaDB."""
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"[RAG] Built vector store with {len(chunks)} chunks.")
    return db


def load_vector_store():
    """Load existing ChromaDB or build it if it doesn't exist."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        print("[RAG] Loaded existing vector store.")
    else:
        db = build_vector_store()
    return db


def retrieve(db, query: str, k: int = 4) -> list[str]:
    """Retrieve top-k relevant chunks for a query."""
    results = db.similarity_search_with_relevance_scores(query, k=k)
    # Filter by relevance threshold
    relevant = [doc.page_content for doc, score in results if score >= 0.3]
    return relevant
