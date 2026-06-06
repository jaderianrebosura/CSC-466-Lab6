import numpy as np
import xlrd

MISSING = 99.0

def load_ratings_xls(filename):
    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_index(0)

    ratings = []

    for r in range(sheet.nrows):
        row = sheet.row_values(r)

        joke_ratings = row[1:101]

        if len(joke_ratings) < 100:
            continue

        joke_ratings = [
            float(value) if value != "" else MISSING
            for value in joke_ratings
        ]

        ratings.append(joke_ratings)

    return np.array(ratings, dtype=float)

def compute_stats(ratings):
    rated_mask = ratings != MISSING

    global_mean = np.mean(ratings[rated_mask])

    user_means = np.zeros(ratings.shape[0])

    for user_index in range(ratings.shape[0]):
        user_ratings = ratings[user_index, :]
        user_rated = user_ratings != MISSING

        if np.any(user_rated):
            user_means[user_index] = np.mean(user_ratings[user_rated])
        else:
            user_means[user_index] = global_mean

    item_means = np.zeros(ratings.shape[1])

    for item_index in range(ratings.shape[1]):
        item_ratings = ratings[:, item_index]
        item_rated = item_ratings != MISSING

        if np.any(item_rated):
            item_means[item_index] = np.mean(item_ratings[item_rated])
        else:
            item_means[item_index] = global_mean

    stats = {
        "rated_mask": rated_mask,
        "global_mean": global_mean,
        "user_means": user_means,
        "item_means": item_means
    }

    return stats


if __name__ == "__main__":
    ratings = load_ratings_xls("jester-data-1.xls")
    stats = compute_stats(ratings)

    print("Shape:", ratings.shape)
    print("Global mean:", stats["global_mean"])

    print("\nFirst 5 user means:")
    print(stats["user_means"][:5])

    print("\nFirst 10 item means:")
    print(stats["item_means"][:10])