import numpy as np
import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # for reproducible results


class FakeNewsEDA:
    def __init__(self, config_path=None):
        """
        Initializes the EDA by reading the config file and dataset.
        If config_path is not provided, it defaults to '../config/config.yaml'.
        """
        # Default config path if none provided
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

        self.config = self.load_config(config_path)
        self.dataset_path = self.config["DATASET_PATH"]
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


    @staticmethod
    def contains_html(s):
        """
        Checks if the string contains HTML-like tags.
        """
        return bool(re.search(r'<[^>]+>', s))

    @staticmethod
    def get_HTMLContentIndices(df):
        """
        Returns a list of indices where the 'text' column contains HTML content.
        Also prints the count of True/False.
        """
        text_series = df['text'].fillna("").astype(str)
        textsScanResults = text_series.apply(FakeNewsEDA.contains_html)
        print(textsScanResults.value_counts())
        html_indices = textsScanResults[textsScanResults].index.tolist()

        title_series = df['title'].fillna("").astype(str)
        titleScanResults = title_series.apply(FakeNewsEDA.contains_html)
        html_indices += titleScanResults[titleScanResults].index.tolist()

        print(titleScanResults.value_counts())
        print(f"Indices with HTML content: {html_indices}")
        print("Number of articles having HTML content is ", len(html_indices))
        return set(html_indices)

    @staticmethod
    def detect_language(text, max_chars=800):
        """
        Detects the language of the given text.
        Truncates the text to 'max_chars' characters.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = text[:max_chars]  # truncate
        try:
            return detect(text)
        except Exception:
            return None

    @staticmethod
    def get_nonEnglishIndices(df):
        """
        Returns a list of indices for rows whose text language is not English.
        It uses a shortened version of the text for faster detection.
        """
        # Create or overwrite a temporary column with truncated text
        short_texts, short_titles = df['text'].fillna("").astype(str).str[:800], df['title'].fillna("").astype(str).str[:800]
        detected_text_langs, detected_title_langs = short_texts.apply(FakeNewsEDA.detect_language), short_titles.apply(FakeNewsEDA.detect_language)
        non_english_indices = df.index[detected_text_langs != 'en'].tolist()
        non_english_indices += df.index[detected_title_langs != 'en'].tolist()

        return set(non_english_indices)

    @staticmethod
    def getReportOfDuplicates(df):
        """
        Returns a dictionary summarizing duplicates in the dataset.
        """
        report = {
            "Number of title duplicates: ": df['title'].duplicated().sum(),
            "Number of text duplicates: ": df['text'].duplicated().sum(),
            "Number of entire rows duplicates: ": df['text'].duplicated().sum()
        }
        return report

    @staticmethod
    def get_textDuplicatesIndices(df):
        """
        Returns a list of indices where the 'text' column is duplicated.
        """
        text_duplicate_indices = df.index[df['text'].duplicated()].tolist()
        return text_duplicate_indices

    def show_pieChart_of_labels_distribution(self):
        """
        Displays a pie chart showing the distribution of labels.
        Each slice is labeled with the class and its corresponding percentage.
        """
        # Count number of fake and real articles
        fake_count = (self.data['label'] == 1).sum()
        real_count = (self.data['label'] == 0).sum()
        counts = [real_count, fake_count]
        labels = ['0', '1']  # Assuming 0 = real and 1 = fake

        total = sum(counts)
        percentages = [count / total * 100 for count in counts]
        slice_labels = [f'{labels[i]}\n{percentages[i]:.2f}%' for i in range(len(labels))]

        fig, ax = plt.subplots()
        ax.pie(counts, labels=slice_labels, startangle=140)
        ax.axis('equal')
        plt.title('Proportion of Real (0) vs Fake (1)')
        plt.show()

    def analyze_word_counts(self):
        """
        Analyzes and prints statistics about the word counts of the 'text' column.
        Also prints the longest article lengths.
        """
        # Drop rows with missing text and calculate word count per article
        texts = self.data['text'].dropna()
        lensOfArticles = texts.apply(lambda x: len(str(x).split()))

        # Show the 10 longest article lengths (excluding the absolute maximum if needed)
        sorted_lengths = np.sort(lensOfArticles)
        print("Longest lengths: ", sorted_lengths[-11:-1])

        # Print descriptive statistics
        print(texts.apply(lambda x: len(str(x).split())).describe())


if __name__ == "__main__":
    # Creating an instance of the FakeNewsEDA class

    eda = FakeNewsEDA()

    # Show overall null counts
    null_count = eda.data.isnull().values.sum()
    print(f"Number of null cells is {null_count}")

    both_null_count = eda.data[eda.data['title'].isnull() & eda.data['text'].isnull()].shape[0]
    print(f"Number of rows where 'title' and 'text' are both null: {both_null_count}")

    # Report duplicates and text duplicate indices
    duplicate_report = eda.getReportOfDuplicates(eda.data)
    print(duplicate_report)
    print("Text duplicate indices:", eda.get_textDuplicatesIndices(eda.data))

    # HTML content detection
    html_indices = eda.get_HTMLContentIndices(eda.data)

    # Language detection (non-English articles)
    non_english_indices = eda.get_nonEnglishIndices(eda.data)
    print(f"Non-English indices: {non_english_indices}.")
    print(f"Number of non-English articles is {len(non_english_indices)}")

    # Analyze word counts and display statistics
    eda.analyze_word_counts()

    # Show the pie chart for labels distribution
    eda.show_pieChart_of_labels_distribution()





