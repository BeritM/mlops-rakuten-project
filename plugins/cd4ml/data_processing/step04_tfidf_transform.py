# module for applying TF-IDF vectorization to text data
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy.sparse import csr_matrix


def apply_tfidf(
    text_series: pd.Series,
    max_features: int = 5000
):
    """
    Applies TF-IDF vectorization to a pandas Series of text data.

    Args:
        text_series (pd.Series): Series containing cleaned text.
        max_features (int): Maximum number of TF-IDF features. Defaults to 5000.

    Returns:
        tuple: (TF-IDF matrix, fitted TfidfVectorizer instance)
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(text_series)
    return X_tfidf, vectorizer
