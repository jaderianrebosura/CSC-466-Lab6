import sys
import statistics

from cf_core import (
    load_ratings_xls,
    compute_stats,
    random_test_cases,
    evaluate_cases,
    print_results,
    print_method_help
)

data = "jester-data-1.xls"


def main():

    if len(sys.argv) == 1:
        print_method_help()
        print()
        print("Usage:")
        print("python EvaluateCFRandom.py Method Size Repeats")
        return

    if len(sys.argv) != 4:
        print("Error: incorrect number of arguments.")
        print("Usage:")
        print("python EvaluateCFRandom.py Method Size Repeats")
        return

    method_id = int(sys.argv[1])
    size = int(sys.argv[2])
    repeats = int(sys.argv[3])
    ratings = load_ratings_xls(data)
    stats = compute_stats(ratings)

    mae_values = []

    for run in range(repeats):

        print()
        print(f"Run {run + 1}")
        print()

        test_cases = random_test_cases(
            ratings,
            size
        )

        results, mae = evaluate_cases(
            method_id,
            ratings,
            stats,
            test_cases
        )

        print_results(results, mae)

        mae_values.append(mae)

    print()
    print("Summary")
    print()

    print("MAE values:")

    for i, mae in enumerate(mae_values, start=1):
        print(f"Run {i}: {mae:.4f}")

    mean_mae = statistics.mean(mae_values)

    if len(mae_values) > 1:
        std_mae = statistics.stdev(mae_values)
    else:
        std_mae = 0.0

    print()
    print(f"Mean MAE: {mean_mae:.4f}")
    print(f"Std Dev MAE: {std_mae:.4f}")


if __name__ == "__main__":
    main()
