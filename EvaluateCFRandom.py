import sys
import numpy as np

from cf_core import *

def print_help_message():
    print("Usage: python3 EvaluateCFRandom.py Method Size Repeats")
    print()
    print("Methods:")
    print("1: Predict User Cosine")
    print("2: Predict User Pearson KNN")
    print("3: Predict User Cosine KNN")
    print("4: Predict Average KNN")

def main():
    if len(sys.argv) == 1:
        print_help_message()
        return

    if len(sys.argv) != 4:
        print_help_message()
        return
    method_id = int(sys.argv[1])
    size = int(sys.argv[2])
    repeats = int(sys.argv[3])

    ratings = load_ratings_xls("jester-data-1.xls")
    stats = compute_stats(ratings)

    maes = []

    for run in range(repeats):
        test_cases = random_test_cases(ratings, size)
        results, mae = evaluate_cases(method_id, ratings, stats, test_cases)
        print_results(results, mae)
        maes.append(mae)

    print()
    print(f"Mean MAE: {np.mean(maes):.4f}")
    print(f"Std Dev MAE: {np.std(maes):.4f}")

if __name__ == "__main__":
    main()