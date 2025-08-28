# main.py

import re
import pandas as pd
import spacy
from langdetect import detect, DetectorFactory

# Fix randomness in langdetect
DetectorFactory.seed = 0

# Load spaCy small English model (run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    """Basic text normalization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)     # replace URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # keep alphanumeric only
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_language(text: str) -> str:
    """Detect language of text; return 'unknown' if fails."""
    try:
        return detect(text)
    except:
        return "unknown"

def spacy_lemmatize(text: str) -> str:
    """Tokenize + lemmatize using spaCy, remove stopwords."""
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def flag_policy_violations(text: str) -> dict:
    """Apply simple rule-based checks for policies."""
    flags = {
        "contains_url": bool(re.search(r"http\S+|www\S+", text)),
        "advertisement": bool(re.search(r"(visit|promo|discount|buy now|call us)", text)),
        "no_visit_rant": "never been" in text or "haven't visited" in text,
        "irrelevant": bool(re.search(r"(iphone|netflix|politics|government)", text)),  # tweak keyword list
    }
    return flags

def preprocess_reviews(input_path: str, output_path: str, sample_size=None):
    # Load data (assumes CSV with at least a 'review_text' column)
    df = pd.read_csv(input_path)
    
    if sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Drop duplicates & missing
    df = df.drop_duplicates(subset=["review_text"]).dropna(subset=["review_text"])
    
    # Clean + normalize
    df["cleaned_text"] = df["review_text"].apply(clean_text)
    
    # Language filter (keep English only)
    df["lang"] = df["cleaned_text"].apply(detect_language)
    df = df[df["lang"] == "en"]
    
    # Lemmatization
    df["lemmatized_text"] = df["cleaned_text"].apply(spacy_lemmatize)
    
    # Metadata features
    df["review_length"] = df["cleaned_text"].apply(lambda x: len(x.split()))
    
    # Apply policy heuristics
    policy_flags = df["cleaned_text"].apply(flag_policy_violations)
    policy_df = pd.DataFrame(policy_flags.tolist())
    df = pd.concat([df, policy_df], axis=1)
    
    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path} (rows: {len(df)})")

if __name__ == "__main__":
    # Example usage (replace input.csv with your dataset)
    preprocess_reviews("data/input/reviews.csv", "data/output/processed_reviews.csv")
