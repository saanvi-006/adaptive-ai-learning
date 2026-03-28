from app.core.adaptive.quiz_engine import build_quiz_from_chunks

def run_quiz_test():
    # 🔹 Fake minimal content (replace later with real PDF chunks)
    chunks = [
        "Method overloading allows multiple methods with same name but different parameters.",
        "It improves code readability and flexibility.",
        "It can be achieved by changing number of arguments or data types.",
    ]

    engine = build_quiz_from_chunks(chunks)

    print("\n===== QUIZ START =====\n")

    for i in range(5):  # only test 5 questions
        q = engine.get_next_question()

        if not q:
            print("No more questions.")
            break

        print(f"Q{i+1}: {q['question']}")
        for opt in q["options"]:
            print(opt)

        # ✅ simulate user always selecting first option
        user_answer = q["options"][0]

        result = engine.submit_answer(
            user_id="test_user",
            selected_answer=user_answer,
            question=q
        )

        print("Correct:", result["is_correct"])
        print("Next Difficulty:", result["next_difficulty"])
        print("Explanation:", result["explanation"])
        print("-" * 50)

    print("\n===== SUMMARY =====")
    print(engine.summary())
    print("Total ques:",len(engine.questions))


if __name__ == "__main__":
    run_quiz_test()