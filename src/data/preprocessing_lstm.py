import numpy as np
import pandas as pd
import yaml
import os
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scripts.eda import FakeNewsEDA

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Preprocessor:
    def __init__(self, config_path=None):
        # Default config path if none provided
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
            print(config_path)
        self.config = self.load_config(config_path)
        self.dataset_path = os.path.join('..', self.config["DATASET_PATH"])
        print(f"Dataset path is: {self.dataset_path}")

        # Read dataset into a DataFrame
        self.data = pd.read_csv(self.dataset_path)
        print(self.data.info())

    @staticmethod
    def load_config(config_path):
        """Loads and returns the YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config


    def remove_null_records(self):
        self.data = self.data.dropna()

        null_count = self.data.isnull().values.sum()
        print(f"Number of null cells after removal is {null_count}")

    def remove_text_duplicates(self):
        # Get duplicate indices for the current data
        duplicate_indices = FakeNewsEDA.get_textDuplicatesIndices(self.data)
        print(f"Removing {len(duplicate_indices)} duplicate rows")
        # Drop the duplicates (keeping the first occurrence)
        self.data.drop(duplicate_indices, inplace=True)
        # Index reset for having sequential indices
        self.data.reset_index(drop=True, inplace=True)

        duplicate_indices = FakeNewsEDA.get_textDuplicatesIndices(self.data)
        print(f"Number of duplicate articles after removal: {len(duplicate_indices)}")

    def remove_html_articles(self):
        html_indices = FakeNewsEDA.get_HTMLContentIndices(self.data)
        print(f"Removing {len(html_indices)} html rows")
        self.data.drop(html_indices, inplace=True)
        # Index reset for having sequential indices
        self.data.reset_index(drop=True, inplace=True)
        html_indices = FakeNewsEDA.get_HTMLContentIndices(self.data)
        print(f"Num of html articles after removal is {len(html_indices)}")

    def remove_nonEnglish_articles(self):
        nonEnglish_indices = FakeNewsEDA.get_nonEnglishIndices(self.data)
        print(f"Removing {len(nonEnglish_indices)} non-english rows")
        self.data.drop(nonEnglish_indices, inplace=True)
        # Index reset for having sequential indices
        self.data.reset_index(drop=True, inplace=True)

        nonEnglish_indices = FakeNewsEDA.get_nonEnglishIndices(self.data)
        print(f"Num of non-english articles after removal: {len(nonEnglish_indices)}.")

    def clean_column(self, column_name):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def clean(text):
            # Expand contractions (e.g., "can't" -> "cannot")
            try:
                text = contractions.fix(text)
            except:
                print(f"The problematic text is {text}")
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation (keep whitespace and word characters)
            text = re.sub(r'[^\w\s]', '', text)
            # Tokenize the text (split on whitespace)
            tokens = text.split()
            # Remove stop words and lemmatize each token
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
            # Rejoin tokens into a cleaned string
            return " ".join(tokens)

        # Apply cleaning function to the 'text' column
        self.data[f'{column_name}_clean'] = self.data[column_name].fillna("").astype(str).apply(clean)
        print(f"{column_name} cleaning completed.")

    def tokenize_and_pad_columns(self, columns, max_lens, num_words=20000):
        """
        Tokenizes and pads multiple columns. The 'columns' argument is a list of column names to process.
        Returns a dictionary with column_name -> padded sequences.
        """
        # We can build a single tokenizer if you want the same vocabulary across columns
        tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")

        # Combine all the text from the specified columns to fit the tokenizer
        combined_texts = []
        for col in columns:
            combined_texts.extend(self.data[col])

        # Fit tokenizer on the combined text
        tokenizer.fit_on_texts(combined_texts)

        padded_sequences_dict = {}
        for col, max_len in zip(columns, max_lens):
            sequences = tokenizer.texts_to_sequences(self.data[col])
            padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
            padded_sequences_dict[col] = padded

        self.tokenizer = tokenizer

        return padded_sequences_dict

    def save_preprocessed_columns(self, padded_dict, labels_column='label', file_path="preprocessed_data_lstm.npz"):
        """
        Saves the padded sequences from different columns and labels into an NPZ file.
        """
        arrays_to_save = {}
        for col, arr in padded_dict.items():
            arrays_to_save[col] = arr
        arrays_to_save['labels'] = self.data[labels_column].values
        np.savez(file_path, **arrays_to_save)

if __name__ == "__main__":
    preprocessedData = Preprocessor()

    preprocessedData.remove_null_records()
    preprocessedData.remove_text_duplicates()
    preprocessedData.remove_html_articles()
    preprocessedData.remove_nonEnglish_articles()
    preprocessedData.clean_column("text")
    preprocessedData.clean_column("title")
    padded_dict = preprocessedData.tokenize_and_pad_columns(["title_clean", "text_clean"], [64, 448])
    preprocessedData.save_preprocessed_columns(padded_dict)
