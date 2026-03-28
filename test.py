from app.core.adaptive.engine import update_user_performance, adapt_response, get_user_performance, reset_user

def run_engine_test():
    user_id = "test_user"

    print("\n===== ENGINE TEST =====\n")

    # Reset user (clean test)
    reset_user(user_id)

    # Step 1: Simulate performance
    update_user_performance(user_id, False, "conceptual")  # wrong
    update_user_performance(user_id, False, "conceptual")  # wrong
    update_user_performance(user_id, True,  "factual")     # correct

    # Step 2: Print stats
    stats = get_user_performance(user_id)

    print("STATS:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Step 3: Test response adaptation
    answer = "Method overloading improves code readability."

    adapted = adapt_response(
        user_id=user_id,
        intent=stats["weak_intents"][0],
        answer=answer
    )

    print("\nADAPTED RESPONSE:")
    print(adapted)


if __name__ == "__main__":
    run_engine_test()