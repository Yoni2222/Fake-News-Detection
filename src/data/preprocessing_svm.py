# preprocessing_svm.py
import os
import numpy as np
import pandas as pd
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from scripts.eda import FakeNewsEDA
from gensim.downloader import load as gensim_load  # we'll load pretrained embeddings


class PreprocessorSVM:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.dataset_path = os.path.join('..', self.config["DATASET_PATH"])

        # Read CSV
        print(f"Reading dataset from: {self.dataset_path}")
        self.data = pd.read_csv(self.dataset_path)
        print(self.data.info())

        print("Loading pretrained embeddings (glove-wiki-gigaword-50)...")
        self.embedding_model = gensim_load("glove-wiki-gigaword-50")
        self.embedding_dim = self.embedding_model.vector_size

    @staticmethod
    def load_config(config_path):
        """Loads and returns the YAML configuration."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def remove_null_records(self):
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        null_count = self.data.isnull().values.sum()
        print(f"Number of null cells after removal is {null_count}")

    def remove_text_duplicates(self):
        duplicate_indices = FakeNewsEDA.get_textDuplicatesIndices(self.data)
        print(f"Removing {len(duplicate_indices)} duplicate rows")
        self.data.drop(duplicate_indices, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def remove_html_articles(self):
        html_indices = FakeNewsEDA.get_HTMLContentIndices(self.data)
        print(f"Removing {len(html_indices)} html rows")
        self.data.drop(html_indices, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def remove_nonEnglish_articles(self):
        nonEnglish_indices = FakeNewsEDA.get_nonEnglishIndices(self.data)
        print(f"Removing {len(nonEnglish_indices)} non-english rows")
        self.data.drop(nonEnglish_indices, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def clean_column(self, column_name):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def clean(text):
            text = contractions.fix(text)
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
            tokens = text.split()
            # Remove stop words + lemmatize
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
            return " ".join(tokens)

        #self.data[f'{column_name}_clean'] = self.data[column_name].fillna("").astype(str).apply(clean)
        #print(f"{column_name} cleaning completed.")
        self.data[f'{column_name}_tokens'] = self.data[column_name].fillna("").astype(str).apply(clean)
        print(f"{column_name} cleaning completed. Tokens stored in '{column_name}_tokens'.")


    def combine_title_text(self, title_col="title_tokens", text_col="text_tokens"):
        """
        Combine lists of tokens from title + text into a single list for each sample.
        """
        combined_tokens = []
        for i in range(len(self.data)):
            title_tokens = self.data.loc[i, title_col] if title_col in self.data else []
            text_tokens = self.data.loc[i, text_col]  if text_col in self.data else []
            # Combine
            combined = title_tokens + text_tokens
            combined_tokens.append(combined)
        self.data['combined_tokens'] = combined_tokens
        print("Combined title + text tokens into 'combined_tokens' column.")


    def create_average_embeddings(self):
        """
        Convert each row's 'combined_tokens' into a single (embedding_dim,) vector
        by averaging the embedding vectors of each token. If a token isn't in the
        pretrained vocabulary, skip it or treat as zero-vector.
        Returns X (shape: [N, embedding_dim]) and y (labels).
        """
        N = len(self.data)
        X = np.zeros((N, self.embedding_dim), dtype=np.float32)
        y = self.data['label'].values

        for i in range(N):
            tokens = self.data.loc[i, 'combined_tokens']
            # Accumulate embeddings
            valid_vecs = []
            for token in tokens:
                if token in self.embedding_model.key_to_index:
                    vec = self.embedding_model[token]
                    valid_vecs.append(vec)
            if len(valid_vecs) == 0:
                # If no valid tokens, you can keep X[i] as zero-vector or random
                pass
            else:
                mean_vec = np.mean(valid_vecs, axis=0)  # shape: (embedding_dim,)
                X[i] = mean_vec

        print(f"Created average embeddings. Shape = {X.shape}")
        return X, y


    def save_embeddings_data(self, X, y, file_path="embeddings_data.npz"):
        """
        Saves the (N, embedding_dim) embeddings array and labels y.
        """
        np.savez(file_path, X=X, y=y)
        print(f"Saved embedding data to {file_path}")



if __name__ == "__main__":
    # Example usage:
    config_path = os.path.join("..", "..", "config", "config.yaml")  # adjust as needed
    preprocessor = PreprocessorSVM(config_path)

    preprocessor.remove_null_records()
    preprocessor.remove_text_duplicates()
    preprocessor.remove_html_articles()
    preprocessor.remove_nonEnglish_articles()

    preprocessor.clean_column("title")
    preprocessor.clean_column("text")

    preprocessor.combine_title_text("title_tokens", "text_tokens")

    X, y = preprocessor.create_average_embeddings()

    # Save to .npz for later usage
    preprocessor.save_embeddings_data(X, y, file_path="embeddings_data.npz")

