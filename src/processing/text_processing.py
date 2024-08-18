import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import numpy as np

# To download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """
    Clean the input text by removing punctuation and stopwords.
    :param text: Raw text
    :return: Cleaned text
    """
    try:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        cleaned_tokens = [token for token in tokens if token not in stop_words]
        cleaned_text = ' '.join(cleaned_tokens)
        logging.info("Text cleaned")
        return cleaned_text
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        raise

def extract_tfidf_features(text, max_features=100):
    """
    Extracting TF-IDF features from text.
    :param text: Cleaned text
    :param max_features: Maximum number of features to extract
    :return: TF-IDF features
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform([text])
        tfidf_features = tfidf_matrix.toarray().flatten()
        logging.info("TF-IDF features extracted")
        return tfidf_features
    except Exception as e:
        logging.error(f"Error extracting TF-IDF features: {e}")
        raise

def process_text(text):
    """
    Processing text input and extracting features.
    :param text: Raw text input
    :return: Extracted text features
    """
    try:
        cleaned_text = clean_text(text)
        tfidf_features = extract_tfidf_features(cleaned_text)
        
        logging.info("Extracted features from text input")
        
        return tfidf_features
    except Exception as e:
        logging.error(f"Error processing text: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python text_processing.py <text_input>")
        sys.exit(1)

    text_input = sys.argv[1]
    features = process_text(text_input)
    print("Extracted Features:\n", features)
