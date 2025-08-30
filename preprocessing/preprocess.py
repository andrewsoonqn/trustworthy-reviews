# main.py

import pandas as pd
import re
import spacy

from spacytextblob.spacytextblob import SpacyTextBlob
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Fix randomness in langdetect
DetectorFactory.seed = 0

# Load spaCy small English model (run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

def detect_language(text: str) -> str:
    """Detect language of text; return 'unknown' if fails."""
    try:
        return detect(text)
    except:
        return "unknown"

def contains_url(text: str) -> bool:
    return "URL" in text.split()

def count_exclamation(text: str) -> int:
    return text.count("!")

def count_all_caps(text: str) -> int:
    return sum(1 for w in text.split() if w.isupper() and len(w) > 1)

def clean_text(text: str) -> str:
    """
    Minimal cleaning for LLM: replace URLs, remove excessive spaces, keep capitalization/punctuation/emojis.
    """
    if not isinstance(text, str):
        return ""

    # Use [^\s] instead of \S to avoid SyntaxWarning
    text = re.sub(r"http[^\s]+|www[^\s]+", " URL ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def sentiment(text: str) -> float:
    doc = nlp(text)
    polarity = doc._.blob.polarity
    return round(polarity, 2)

def subjectivity(text: str) -> float:
    doc = nlp(text)
    subjectivity = doc._.blob.subjectivity
    return round(subjectivity, 2)

def flag_duplicate_reviews(df, similarity_threshold=0.75):
    df["duplicate_flag"] = False

    # Group by business_name to only compare reviews within the same restaurant
    for business, group in df.groupby("business_name"):
        if len(group) < 2:
            continue  # no comparisons needed

        # Vectorize the cleaned reviews for this business
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(group["cleaned_text"])

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Flag reviews that are very similar to any other review
        duplicate_indices = set()
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if similarity_matrix[i, j] >= similarity_threshold:
                    duplicate_indices.add(group.index[i])
                    duplicate_indices.add(group.index[j])

        # Mark flagged reviews
        df.loc[list(duplicate_indices), "duplicate_flag"] = True

    return df

def preprocess_reviews(input_path: str, output_path: str, sample_size=None):
    df = pd.read_csv(input_path)

    if sample_size:
        df = df.sample(sample_size, random_state=42)

    # Drop duplicates & missing
    df = df.drop_duplicates(subset=["review_text"]).dropna(subset=["review_text"])

    # Clean + normalize
    df["cleaned_text"] = df["review_text"].apply(clean_text)

    df["sentiment"] = df["cleaned_text"].apply(sentiment)
    df["subjectivity"] = df["cleaned_text"].apply(subjectivity)

    # Language filter (keep English only)
    df["lang"] = df["cleaned_text"].apply(detect_language)
    df = df[df["lang"] == "en"]

    # Metadata features
    df["review_length"] = df["cleaned_text"].apply(lambda x: len(x.split()))
    df["exclaim_count"] = df["cleaned_text"].apply(count_exclamation)
    df["caps_count"] = df["cleaned_text"].apply(count_all_caps)
    df["contains_url"] = df["cleaned_text"].apply(contains_url)

    # duplicate reviews flagger
    """df = flag_duplicate_reviews(df)"""

    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path} (rows: {len(df)})")
    return df

    return df

if __name__ == "__main__":
    df = preprocess_reviews("data/input/reviews.csv", "data/output/processed_reviews_500_new.csv", 500)
    llm.compile_reviews(df, "data/llmOutput/llmevaluated_reviews_Kaggle_500.csv")