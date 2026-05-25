from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def _load_text_documents(data_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(data_dir),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        documents.extend(loader.load())
    return documents


def _load_pdf_documents(data_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for pdf_path in data_dir.glob("**/*.pdf"):
        documents.extend(PyPDFLoader(str(pdf_path)).load())
    return documents


def load_documents(data_dir: Path = settings.data_dir) -> list[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents = _load_text_documents(data_dir)
    documents.extend(_load_pdf_documents(data_dir))

    if not documents:
        raise ValueError(f"No .txt, .md, or .pdf documents found in {data_dir}")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)

