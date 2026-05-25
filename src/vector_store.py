import shutil

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import settings


def get_embeddings():
    if settings.llm_provider == "openai":
        return OpenAIEmbeddings(model=settings.openai_embedding_model)

    return GoogleGenerativeAIEmbeddings(model=settings.google_embedding_model)


def build_vector_store(reset: bool = True) -> Chroma:
    from src.loaders import load_documents, split_documents

    if reset and settings.persist_dir.exists():
        shutil.rmtree(settings.persist_dir)

    documents = load_documents()
    chunks = split_documents(documents)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=str(settings.persist_dir),
    )


def load_vector_store() -> Chroma:
    if not settings.persist_dir.exists():
        raise FileNotFoundError(
            "Vector database not found. Run `python ingest.py` before asking questions."
        )

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.persist_dir),
    )
