import pandas as pd

def count_flags_df(df: pd.DataFrame) -> dict:
    """
    Counts total flags for spam, irrelevant, and rant in a DataFrame,
    as well as how many reviews were flagged at least once.

    Parameters:
        df (pd.DataFrame): DataFrame containing "spam", "irrelevant", and "rant" columns.

    Returns:
        dict: Dictionary with counts of each flag and totals.
    """
    flag_columns = ["inauthentic", "irrelevant", "advertisement", "rant"]

    counts = {col: int(df[col].sum()) if col in df else 0 for col in flag_columns}

    # Count reviews with at least one flag
    if all(col in df for col in flag_columns):
        reviews_flagged = (df[flag_columns].sum(axis=1) > 0).sum()
    else:
        reviews_flagged = 0

    totals = {
        **counts,
        "total_flags": sum(counts.values()),
        "reviews_with_flags": int(reviews_flagged)
    }

    return totals


def count_flags_csv(csv_path: str) -> dict:
    """
    Reads a CSV of evaluated reviews and counts total flags for
    inauthentic, irrelevant, advertisement, and rant,
    as well as how many reviews were flagged at least once.

    Parameters:
        csv_path (str): Path to the evaluated reviews CSV file.

    Returns:
        dict: Dictionary with counts of each flag and totals.
    """
    df = pd.read_csv(csv_path)

    flag_columns = ["inauthentic", "irrelevant", "advertisement", "rant"]

    counts = {col: int(df[col].sum()) if col in df else 0 for col in flag_columns}

    # Count reviews with at least one flag
    if all(col in df for col in flag_columns):
        reviews_flagged = (df[flag_columns].sum(axis=1) > 0).sum()
    else:
        reviews_flagged = 0

    totals = {
        **counts,
        "total_flags": sum(counts.values()),
        "reviews_with_flags": int(reviews_flagged)
    }

    return totals

if __name__ == "__main__":
    counts = count_flags_csv("data/llm_output/llmevaluated_reviews_Kaggle_400.csv")
    print("Flag summary:", counts)
