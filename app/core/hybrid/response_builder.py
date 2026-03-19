def build_response(query: str, intent: str, answer: str, context: list):

    return {
        "answer": answer,
        "source": "RAG",
        "explanation": "Generated using retrieved study material.",
    }