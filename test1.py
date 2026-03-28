from app.core.rag.pipeline import run_rag_pipeline

def test_intent_pipeline():
    source = "data/uploads/sample.pdf"

    print("\n===== INTENT + RAG TEST =====\n")

    queries = [
        "What is method overloading?",
        "Why do we use method overloading?",
        "Explain garbage collection in Java",
    ]

    for i, query in enumerate(queries, 1):
        result = run_rag_pipeline(query=query, source=source)

        print(f"Q{i}: {query}")
        print("Detected Intent:", result["intent"])
        print("Answer:")
        print(result["answer"])
        print("-" * 60)


if __name__ == "__main__":
    test_intent_pipeline()