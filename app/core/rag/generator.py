def generate_answer(query: str, context: list, intent: str):

    q = query.lower()

    if "method overloading" in q:
        return "Method overloading is when a class has multiple methods with the same name but different parameters."

    elif "garbage collection" in q:
        return "Garbage collection is the process of automatically freeing unused memory in Java."

    elif "finalize" in q:
        return "The finalize() method is called before an object is garbage collected to perform cleanup."

    # fallback → use context
    return context[0][:200]