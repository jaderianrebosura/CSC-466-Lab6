import random
import numpy as np
import xlrd

MISSING = 99.0
MIN_RATING = -10.0
MAX_RATING = 10.0
DEFAULT_K = 20


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

    return {
        "rated_mask": rated_mask,
        "global_mean": global_mean,
        "user_means": user_means,
        "item_means": item_means
    }


def clamp(value):
    if value < MIN_RATING:
        return MIN_RATING
    elif value > MAX_RATING:
        return MAX_RATING
    else:
        return value


def cosine_similarity_user(ratings, user_a, user_b):
    ratings_a = ratings[user_a, :]
    ratings_b = ratings[user_b, :]

    common_mask = (ratings_a != MISSING) & (ratings_b != MISSING)

    if not np.any(common_mask):
        return 0.0

    vector_a = ratings_a[common_mask]
    vector_b = ratings_b[common_mask]

    numerator = np.dot(vector_a, vector_b)
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def pearson_similarity_user(ratings, user_a, user_b):
    ratings_a = ratings[user_a, :]
    ratings_b = ratings[user_b, :]

    common_mask = (ratings_a != MISSING) & (ratings_b != MISSING)

    if np.sum(common_mask) < 2:
        return 0.0

    a = ratings_a[common_mask]
    b = ratings_b[common_mask]

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    a_centered = a - a_mean
    b_centered = b - b_mean

    numerator = np.dot(a_centered, b_centered)
    denominator = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def get_neighbors(ratings, user_index, item_index, similarity_function, k=None):
    neighbors = []

    for other_user in range(ratings.shape[0]):
        if other_user == user_index:
            continue

        other_rating = ratings[other_user, item_index]

        if other_rating == MISSING:
            continue

        similarity = similarity_function(ratings, user_index, other_user)

        if similarity <= 0:
            continue

        neighbors.append((similarity, other_user, other_rating))

    neighbors.sort(key=lambda x: x[0], reverse=True)

    if k is not None:
        neighbors = neighbors[:k]

    return neighbors


def predict_item_average(_ratings, stats, _user_id, item_id):
    item_index = item_id - 1
    return clamp(stats["item_means"][item_index])


def predict_user_cosine(ratings, stats, user_id, item_id):
    user_index = user_id - 1
    item_index = item_id - 1

    neighbors = get_neighbors(
        ratings,
        user_index,
        item_index,
        cosine_similarity_user,
        k=None
    )

    if len(neighbors) == 0:
        return stats["item_means"][item_index]

    numerator = 0.0
    denominator = 0.0

    for similarity, _other_user, other_rating in neighbors:
        numerator += similarity * other_rating
        denominator += abs(similarity)

    if denominator == 0:
        return stats["item_means"][item_index]

    return clamp(numerator / denominator)


def predict_user_pearson_knn(ratings, stats, user_id, item_id, k=DEFAULT_K):
    user_index = user_id - 1
    item_index = item_id - 1

    neighbors = get_neighbors(
        ratings,
        user_index,
        item_index,
        pearson_similarity_user,
        k=k
    )

    if len(neighbors) == 0:
        return stats["item_means"][item_index]

    numerator = 0.0
    denominator = 0.0

    target_mean = stats["user_means"][user_index]

    for similarity, neighbor_index, neighbor_rating in neighbors:
        neighbor_mean = stats["user_means"][neighbor_index]

        numerator += similarity * (neighbor_rating - neighbor_mean)
        denominator += abs(similarity)

    if denominator == 0:
        return target_mean

    return clamp(target_mean + numerator / denominator)


def predict_user_cosine_knn(ratings, stats, user_id, item_id, k=DEFAULT_K):
    user_index = user_id - 1
    item_index = item_id - 1

    neighbors = get_neighbors(
        ratings,
        user_index,
        item_index,
        cosine_similarity_user,
        k=k
    )

    if len(neighbors) == 0:
        return stats["item_means"][item_index]

    numerator = 0.0
    denominator = 0.0

    for similarity, _other_user, other_rating in neighbors:
        numerator += similarity * other_rating
        denominator += abs(similarity)

    if denominator == 0:
        return stats["item_means"][item_index]

    return clamp(numerator / denominator)


def predict_average_knn(ratings, stats, user_id, item_id, k=DEFAULT_K):
    user_index = user_id - 1
    item_index = item_id - 1

    neighbors = get_neighbors(
        ratings,
        user_index,
        item_index,
        cosine_similarity_user,
        k=k
    )

    if len(neighbors) == 0:
        return stats["item_means"][item_index]

    total = 0.0

    for _similarity, _other_user, other_rating in neighbors:
        total += other_rating

    return clamp(total / len(neighbors))


METHODS = {
    0: predict_item_average,
    1: predict_user_cosine,
    2: predict_user_pearson_knn,
    3: predict_user_cosine_knn,
    4: predict_average_knn
}


def predict(method_id, ratings, stats, user_id, item_id):
    user_index = user_id - 1
    item_index = item_id - 1

    original_rating = ratings[user_index, item_index]

    if original_rating != MISSING:
        ratings[user_index, item_index] = MISSING

    try:
        if method_id not in METHODS:
            raise ValueError(f"Unknown method id: {method_id}")

        prediction_function = METHODS[method_id]
        prediction = prediction_function(ratings, stats, user_id, item_id)

    finally:
        ratings[user_index, item_index] = original_rating

    return clamp(prediction)


def random_test_cases(ratings, size):
    test_cases = []

    while len(test_cases) < size:
        user_index = random.randint(0, ratings.shape[0] - 1)
        item_index = random.randint(0, ratings.shape[1] - 1)

        if ratings[user_index, item_index] != MISSING:
            test_cases.append((user_index + 1, item_index + 1))

    return test_cases


def evaluate_cases(method_id, ratings, stats, test_cases):
    total_error = 0.0
    results = []

    for user_id, item_id in test_cases:
        actual = ratings[user_id - 1, item_id - 1]

        predicted = predict(
            method_id,
            ratings,
            stats,
            user_id,
            item_id
        )

        delta = actual - predicted
        total_error += abs(delta)

        results.append(
            (user_id, item_id, actual, predicted, delta)
        )

    mae = total_error / len(test_cases)

    return results, mae


def recommendation_metrics(results):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for _user_id, _item_id, actual, predicted, _delta in results:
        actual_recommend = actual >= 5
        predicted_recommend = predicted >= 5

        if predicted_recommend and actual_recommend:
            tp += 1
        elif predicted_recommend and not actual_recommend:
            fp += 1
        elif not predicted_recommend and not actual_recommend:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


def print_results(results, mae):
    print("userID,itemID,Actual_Rating,Predicted_Rating,Delta_Rating")

    for user_id, item_id, actual, predicted, delta in results:
        print(
            f"{user_id},"
            f"{item_id},"
            f"{actual:.4f},"
            f"{predicted:.4f},"
            f"{delta:.4f}"
        )

    metrics = recommendation_metrics(results)

    print()
    print(f"MAE: {mae:.4f}")
    print()
    print("Confusion Matrix:")
    print("                 Actual Recommend   Actual Do Not Recommend")
    print(f"Pred Recommend   {metrics['tp']}                  {metrics['fp']}")
    print(f"Pred No Rec      {metrics['fn']}                  {metrics['tn']}")
    print()
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    ratings = load_ratings_xls("jester-data-1.xls")
    stats = compute_stats(ratings)

    test_cases = random_test_cases(ratings, size=100)

    results, mae = evaluate_cases(
        method_id=4,
        ratings=ratings,
        stats=stats,
        test_cases=test_cases
    )

    print_results(results, mae)