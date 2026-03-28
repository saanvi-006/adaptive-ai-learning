from app.core.rag.pipeline import get_all_chunks
from app.core.adaptive.flashcard_engine import generate_flashcards

def run_flashcard_test():
    source = "data/uploads/sample.pdf"  # your PDF path

    print("\n===== FLASHCARD TEST =====\n")

    # Step 1: Get chunks
    chunks = get_all_chunks(source)

    print(f"Chunks loaded: {len(chunks)}")

    # Step 2: Generate flashcards
    flashcards = generate_flashcards(chunks, target_count=5)

    # Step 3: Print results
    print(f"\nTotal Flashcards Generated: {len(flashcards)}\n")

    for i, card in enumerate(flashcards, 1):
        print(f"Card {i}")
        print("Q:", card["question"])
        print("A:", card["answer"])
        print("-" * 50)


if __name__ == "__main__":
    run_flashcard_test()