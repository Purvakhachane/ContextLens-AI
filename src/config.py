from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path = BASE_DIR / "data"
    persist_dir: Path = BASE_DIR / "chroma_db"
    collection_name: str = "rag_from_scratch"
    llm_provider: str = os.getenv("LLM_PROVIDER", "google").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    google_embedding_model: str = os.getenv(
        "GOOGLE_EMBEDDING_MODEL",
        "gemini-embedding-001",
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))

    @property
    def has_valid_openai_key(self) -> bool:
        key = self.openai_api_key.strip()
        return bool(key) and "your_openai_key_here" not in key and key.startswith("sk-")

    @property
    def has_valid_google_key(self) -> bool:
        key = self.google_api_key.strip()
        return bool(key) and "your_gemini_key_here" not in key and key.startswith("AIza")

    @property
    def has_valid_provider_key(self) -> bool:
        if self.llm_provider == "openai":
            return self.has_valid_openai_key
        return self.has_valid_google_key


settings = Settings()
