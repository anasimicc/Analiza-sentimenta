import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    """Lightweight cleaning: remove HTML, lowercase, remove urls, punctuation, digits, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)                      # remove html tags
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)    # remove urls
    # replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\d+", " ", text)                        # remove digits
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords_and_short(tokens):
    """Remove stop words and short tokens"""
    stopwords = ENGLISH_STOP_WORDS
    return [t for t in tokens if t not in stopwords and len(t) > 1]

def preprocess_series(series):
    """
    Preprocess a pandas Series of text data
    
    Steps:
    1. Clean text (remove HTML, punctuation, etc.)
    2. Tokenize (split into words)
    3. Remove stop words and short words
    4. Join back into string
    """
    cleaned = series.fillna("").map(clean_text)
    tokenized = cleaned.map(lambda s: s.split())
    filtered = tokenized.map(remove_stopwords_and_short)
    return filtered.map(lambda tokens: " ".join(tokens))

def preprocess_single_text(text):
    """
    Preprocess a single text string
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Cleaned and processed text
    """
    if not isinstance(text, str):
        return ""
    
    # Clean text
    cleaned = clean_text(text)
    
    # Tokenize
    tokens = cleaned.split()
    
    # Remove stop words and short words
    filtered_tokens = remove_stopwords_and_short(tokens)
    
    # Join back
    return " ".join(filtered_tokens)

def get_preprocessing_stats(original_series, processed_series):
    """
    Get statistics about the preprocessing results
    
    Args:
        original_series: Original text data
        processed_series: Preprocessed text data
        
    Returns:
        dict: Statistics about the preprocessing
    """
    stats = {
        'original_avg_length': original_series.str.len().mean(),
        'processed_avg_length': processed_series.str.len().mean(),
        'original_word_count': original_series.str.split().str.len().mean(),
        'processed_word_count': processed_series.str.split().str.len().mean(),
        'reduction_ratio': 1 - (processed_series.str.len().mean() / original_series.str.len().mean())
    }
    return stats
