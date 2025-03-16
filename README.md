# DL_Project - Fake News Detection

# Overview
Fake news detection has become an important challenge in today's digital landscape. This project implements **Machine Learning (SVM)** and **Deep Learning (LSTM)** models to classify news articles as real or fake. The models are trained using a dataset of news articles and evaluated based on accuracy and other performance metrics.

## Features
- **Data Preprocessing**: Cleans and prepares textual data for training.
- **Exploratory Data Analysis (EDA)**: Generates insights from the dataset.
- **Support Vector Machine (SVM)**: A classic machine learning model for text classification.
- **Long Short-Term Memory (LSTM)**: A deep learning-based approach for sequence learning.


Clone the repository:
    git clone https://github.com/your-username/DL_Project.git

How to Run the Project:
    Run Exploratory Data Analysis (EDA):
        python scripts/eda.py
    Preprocess the data:
        python src/data/preprocessing_svm.py
        python src/data/preprocessing_lstm.py
    Train the Models:
        python scripts/train_svm.py
        python scripts/train_lstm.py

Dataset Used: WELFake Dataset (https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)
Contributors: Yonatan Ben Avraham
Results:
    SVM (Kernel = 'rbf') achieved accuracy of 74%
    Best model of LSTM achieved accuracy of 97.5%
Technologies Used
    Python
    Scikit-learn (for ML models)
    TensorFlow/Keras (for Deep Learning)
    Pandas, NumPy
    Matplotlib

This repository includes a detailed academic-style summary of the project. The document, "Fake News Detection.docx", provides an in-depth explanation of the problem, methodologies (SVM and LSTM models), experimental setup, results, conclusions and more.
