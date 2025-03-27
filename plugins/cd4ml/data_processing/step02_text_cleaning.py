import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required resources once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Get stopwords for English and French and also custom stopwords
stop_words_eng = set(stopwords.words('english'))
stop_words_fr = set(stopwords.words('french'))
custom_stopwords = set(["chez", "der", "plu", "haut", "peut", "non", "100", "produit", "lot", "tout", "cet", "cest", "sou", "san"])
stop_words = stop_words_eng.union(stop_words_fr).union(custom_stopwords)

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Cleans a text string by removing special characters (keeping letters, numbers, and spaces),
    converting to lowercase, tokenizing, lemmatizing, and removing stopwords.
    
    Args:
        text (str): Input text string.
    
    Returns:
        str: Cleaned and lemmatized text string.
    """
    # Remove special characters (retain letters, numbers, spaces)
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text if text is not None else '')
    # Convert to lowercase
    cleaned = cleaned.lower()
    # Tokenize the text
    tokens = word_tokenize(cleaned)
    # Lemmatize tokens
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    # Remove stopwords
    filtered_tokens = [token for token in lemmas if token not in stop_words]
    return ' '.join(filtered_tokens)


def process_text_dataframe(df: pd.DataFrame,
                           designation_col: str = 'designation',
                           description_col: str = 'description') -> pd.DataFrame:
    """
    Processes the text data in a DataFrame by merging the 'designation' and 'description'
    columns, cleaning the text, tokenizing, lemmatizing, and removing stopwords.
    Produces a new column 'lemmatized_text' and drops unnecessary columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame with raw text data.
        designation_col (str): Column name for designation. Default is 'designation'.
        description_col (str): Column name for description. Default is 'description'.
    
    Returns:
        pd.DataFrame: Processed DataFrame with a new column 'lemmatized_text'.
    """
    # Merge 'designation' and 'description', handling NaN in 'description'
    df['text'] = df[designation_col] + ' ' + df[description_col].fillna('')
    
    # Remove special characters (keeping letters, numbers, and spaces)
    df['cleaned_text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x) if pd.notnull(x) else ''))
    
    # Convert cleaned text to lowercase
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    
    # Apply tokenization, lemmatization, and stopword removal to produce 'lemmatized_text'
    df['lemmatized_text'] = df['cleaned_text'].apply(
        lambda x: ' '.join([word for word in 
                            [lemmatizer.lemmatize(token) for token in word_tokenize(x.lower())]
                            if word not in stop_words])
    )
    
    # Drop columns that are no longer needed
    drop_cols = [designation_col, description_col, 'productid', 'text']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    return df
