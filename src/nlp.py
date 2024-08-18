import os
import re
import nltk
import spacy
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spacy.lang.en import English

from src.config import Config
from src.utils.file_utils import save_to_file, read_from_file

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

class NLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = English().Defaults.create_tokenizer(nlp)

    def clean_text(self, text):
        """
        Cleaning input text by removing non-alphabetic characters and lowercasing.
        :param text: The input text
        :return: Cleaned text
        """
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.strip()
        return text

    def tokenize_text(self, text):
        """
        To tokenize input text.
        :param text: The input text
        :return: List of tokens
        """
        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens):
        """
        Removing stopwords from token list.
        :param tokens: List of tokens
        :return: List of tokens without stopwords
        """
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens

    def lemmatize_tokens(self, tokens):
        """
        To lemmatize input tokens.
        :param tokens: List of tokens
        :return: List of lemmatized tokens
        """
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def process_text(self, text):
        """
        Processing input text by cleaning, tokenizing, removing stopwords, and lemmatizing.
        :param text: The input text
        :return: Processed text
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        tokens = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        return ' '.join(lemmatized_tokens)

    def spacy_tokenize(self, text):
        """
        Tokenizing the input text using Spacy.
        :param text: The input text
        :return: List of tokens
        """
        doc = nlp(text)
        return [token.text for token in doc]

    def spacy_lemmatize(self, tokens):
        """
        To lemmatize input tokens using Spacy.
        :param tokens: List of tokens
        :return: List of lemmatized tokens
        """
        doc = nlp(' '.join(tokens))
        return [token.lemma_ for token in doc]

    def spacy_remove_stopwords(self, tokens):
        """
        Removing stopwords from the token list using Spacy.
        :param tokens: List of tokens
        :return: List of tokens without stopwords
        """
        return [token for token in tokens if not nlp.vocab[token].is_stop]

if __name__ == "__main__":
    nlp_processor = NLPProcessor()

    example_text = "This is an example sentence to demonstrate the NLP processing capabilities."

    processed_text = nlp_processor.process_text(example_text)
    print(f"Processed text (NLTK): {processed_text}")

    tokens = nlp_processor.spacy_tokenize(example_text)
    tokens = nlp_processor.spacy_remove_stopwords(tokens)
    lemmatized_tokens = nlp_processor.spacy_lemmatize(tokens)
    print(f"Processed text (Spacy): {' '.join(lemmatized_tokens)}")

    save_to_file(processed_text, os.path.join(Config.PROCESSED_DATA_DIR, 'processed_text.txt'))
    print(f"Processed text saved to {Config.PROCESSED_DATA_DIR}/processed_text.txt")
