import sys
import numpy as np
from cf_core import *

def print_help_message():
    print("Usage: python3 EvaluateCFList.py Method Filename")
    print()
    print("Methods:")
    print("0: Predict Item Average")
    print("1: Predict User Cosine")
    print("2: Predict User Pearson KNN")
    print("3: Predict User Cosine KNN")
    print("4: Predict Average KNN")

def load_test_cases(filename):
    test_cases = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            user, item = line.split(",")
            test_cases.append((int(user), int(item)))
    return test_cases

def main():
    if len(sys.argv) != 3:
        print_help_message()
        return
    method_id = int(sys.argv[1])
    filename = sys.argv[2]

    ratings = load_ratings_xls("jester-data-1.xls")
    stats = compute_stats(ratings)

    test_cases = load_test_cases(filename)

    results, mae = evaluate_cases(method_id, ratings, stats, test_cases)

    print_results(results, mae)

if __name__ == "__main__":
    main()