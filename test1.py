from app.core.rag.pipeline import get_all_chunks
from app.core.adaptive.quiz_engine import build_quiz_from_chunks
from app.core.adaptive.engine import reset_user, get_user_performance

def run_quiz_adaptive_test():
    user_id = "test_user"
    source = "data/uploads/sample.pdf"

    print("\n===== QUIZ + ADAPTIVE FINAL TEST =====\n")

    # reset user
    reset_user(user_id)

    # STEP 1: Load chunks
    chunks = get_all_chunks(source)

    # STEP 2: Build quiz
    quiz = build_quiz_from_chunks(chunks)

    print(f"Total Questions Generated: {len(quiz.questions)}")

    # STEP 3: Simulate realistic user performance
    # Pattern: wrong → wrong → correct → correct → correct → wrong → correct...
    performance_pattern = [False, False, True, True, True, False, True, False, True, True]

    for i, is_correct in enumerate(performance_pattern):
        q = quiz.get_next_question()
        if not q:
            break

        print(f"\nQ{i+1}: {q['question']}")
        print("Difficulty:", q["difficulty"])

        # simulate answer
        user_answer = q["correct_answer"] if is_correct else "wrong_option"

        result = quiz.submit_answer(user_id, user_answer, q)

        print("Correct:", result["is_correct"])
        print("Next Difficulty:", result["next_difficulty"])
        print("Explanation:", result["explanation"])

        print("-" * 50)

    # STEP 4: Final performance stats
    stats = get_user_performance(user_id)

    print("\n===== FINAL USER STATS =====")
    print(stats)

    print("\n===== QUIZ SUMMARY =====")
    print(quiz.summary())


if __name__ == "__main__":
    run_quiz_adaptive_test()