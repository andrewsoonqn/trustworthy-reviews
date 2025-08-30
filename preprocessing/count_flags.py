import json


def count_flags(json_path):
    """
    Reads a JSON file of reviews and counts total flags for spam, irrelevant, and rant,
    as well as how many reviews were flagged at least once.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    spam_count = sum(r.get("spam", 0) for r in reviews)
    irrelevant_count = sum(r.get("irrelevant", 0) for r in reviews)
    rant_count = sum(r.get("rant", 0) for r in reviews)

    # Count reviews with at least one flag
    reviews_flagged = sum(
        1 for r in reviews if r.get("spam", 0) or r.get("irrelevant", 0) or r.get("rant", 0)
    )

    totals = {
        "spam": spam_count,
        "irrelevant": irrelevant_count,
        "rant": rant_count,
        "total_flags": spam_count + irrelevant_count + rant_count,
        "reviews_with_flags": reviews_flagged
    }

    return totals

if __name__ == "__main__":
    counts = count_flags("data/llmOutput/llmevaluated_reviews_Kaggle_400.json")
    print("Flag summary:", counts)