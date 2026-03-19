def route_query(query: str):
    q = query.lower()

    if "why" in q or "how" in q or "explain" in q:
        return "conceptual"
    elif "quiz" in q or "flashcard" in q:
        return "learning"
    else:
        return "factual"