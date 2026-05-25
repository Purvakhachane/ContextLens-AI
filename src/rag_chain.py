from statistics import mean

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.config import settings
from src.vector_store import load_vector_store


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are ContextLens AI, a careful retrieval assistant. Answer only from the provided context. "
            "If the answer is not in the context, say you do not know. Include concise source citations. "
            "Use the requested answer style without inventing facts.",
        ),
        (
            "human",
            "Answer style: {style}\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:",
        ),
    ]
)


def format_documents(docs) -> str:
    formatted = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        label = f"{source}, page {page + 1}" if page is not None else source
        formatted.append(f"[Source {index}: {label}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def retrieve(question: str):
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retrieval_k})
    return retriever.invoke(question)


def retrieve_with_scores(question: str):
    vector_store = load_vector_store()
    return vector_store.similarity_search_with_relevance_scores(
        question,
        k=settings.retrieval_k,
    )


def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def answer_question(question: str, style: str = "Balanced") -> dict:
    scored_docs = retrieve_with_scores(question)
    docs = [doc for doc, _score in scored_docs]
    scores = [max(0, min(1, score)) for _doc, score in scored_docs]
    confidence = mean(scores) if scores else 0
    context = format_documents(docs)
    if settings.llm_provider == "openai":
        llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    else:
        llm = ChatGoogleGenerativeAI(model=settings.google_model, temperature=0)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context, "style": style})
    return {
        "answer": answer,
        "sources": docs,
        "scored_sources": scored_docs,
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
    }
