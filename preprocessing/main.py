# main.py

import re
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from langdetect import detect, DetectorFactory

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

def sentiment(text: str) -> str:
    doc = nlp(text)
    polarity = doc._.blob.polarity
    """
    if polarity >= 0.05:
        sentiment_label = "positive"
    elif polarity <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    return sentiment_label
    """
    return round(polarity, 2)

def subjectivity(text: str) -> str:
    doc = nlp(text)
    subjectivity = doc._.blob.subjectivity
    """
    if subjectivity >= 0.05:
        subjectivity_label = "positive"
    elif subjectivity <= -0.05:
        subjectivity_label = "negative"
    else:
        subjectivity_label = "neutral"
    return subjectivity_label
    """
    return round(subjectivity, 2)


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
    
    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path} (rows: {len(df)})")

if __name__ == "__main__":
    preprocess_reviews("data/input/reviews.csv", "data/output/processed_reviews.csv")
