import argparse

from src.rag_chain import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question using the RAG system.")
    parser.add_argument("question", help="Question to ask the indexed documents")
    parser.add_argument(
        "--style",
        choices=["Balanced", "Brief", "Detailed", "Executive"],
        default="Balanced",
        help="Answer style to use",
    )
    args = parser.parse_args()

    result = answer_question(args.question, style=args.style)
    print("\nAnswer\n------")
    print(result["answer"])
    print(f"\nConfidence: {result['confidence_label']} ({result['confidence']:.0%})")
    print("\nRetrieved Sources\n-----------------")
    for doc, score in result["scored_sources"]:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        page_text = f", page {page + 1}" if page is not None else ""
        print(f"- {source}{page_text} | relevance {score:.0%}")


if __name__ == "__main__":
    main()
