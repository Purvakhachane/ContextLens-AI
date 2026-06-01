from src.vector_store import build_vector_store

def main() -> None:
    vector_store = build_vector_store(reset=True)
    count = vector_store._collection.count()
    print(f"Indexed {count} chunks into the local vector database.")


if __name__ == "__main__":
    main()
