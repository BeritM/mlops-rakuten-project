# module for text cleaning
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required resources once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Cleans a string by removing non-word characters, converting to lowercase,
    removing stopwords, and applying lemmatization.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned and lemmatized text string.
    """
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmas)