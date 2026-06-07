import sys
import csv

from cf_core import (
    load_ratings_xls,
    compute_stats,
    evaluate_cases,
    print_results,
    print_method_help,
    MISSING
)

data = "jester-data-1.xls"


def load_test_cases(filename, ratings):
    test_cases = []

    with open(filename, "r") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) < 2:
                continue

            user_id = int(row[0])
            item_id = int(row[1])

            user_index = user_id - 1
            item_index = item_id - 1

            if user_index < 0 or user_index >= ratings.shape[0]:
                continue

            if item_index < 0 or item_index >= ratings.shape[1]:
                continue

            if ratings[user_index, item_index] == MISSING:
                continue

            test_cases.append((user_id, item_id))

    return test_cases


def main():
    if len(sys.argv) == 1:
        print_method_help()
        print()
        print("Usage:")
        print("python EvaluateCFList.py Method Filename")
        return

    if len(sys.argv) != 3:
        print("Error: incorrect number of arguments.")
        print("Usage:")
        print("python EvaluateCFList.py Method Filename")
        return

    method_id = int(sys.argv[1])
    filename = sys.argv[2]

    ratings = load_ratings_xls(data)
    stats = compute_stats(ratings)

    test_cases = load_test_cases(filename, ratings)

    results, mae = evaluate_cases(
        method_id,
        ratings,
        stats,
        test_cases
    )

    print_results(results, mae)


if __name__ == "__main__":
    main()
