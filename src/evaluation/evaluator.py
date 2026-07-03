"""
ForgeGuard Evaluation Module
"""

def run_evaluation(engine, save_dir="outputs"):

    print("=" * 60)
    print("ForgeGuard Evaluation")
    print("=" * 60)

    questions = [
        "What is machine learning?",
        "What is deep learning?",
        "What is a neural network?"
    ]

    results = []

    for question in questions:

        print(f"\nQuestion: {question}")

        response = engine["ask"](question)

        print(f"\nResponse:\n{response}")

        results.append(
            {
                "question": question,
                "response": response
            }
        )

    metrics = {
        "accuracy": 0.0,
        "hallucination_rate": 0.0,
        "ece": 0.0,
        "num_examples": len(results)
    }

    print("\nEvaluation Complete")

    return metrics