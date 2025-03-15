import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
def split_train_test(preprocessed_data):
    # Load the preprocessed NPZ file

    # Retrieve arrays from the NPZ file
    text_clean, title_clean, labels = preprocessed_data["text_clean"], preprocessed_data["title_clean"], preprocessed_data["labels"]
    print("len of labels is ", len(labels))
    N = len(labels)
    cutoff = int(0.9 * N)

    texts, titles, labels_cut = text_clean[:cutoff], title_clean[:cutoff], labels[:cutoff]

    train_title, test_title, train_text, test_text, y_train, y_test = train_test_split(
        titles, texts, labels_cut, test_size=0.2, random_state=42
    )
    print("len of labels is ", len(y_train))
    print("len of train_title is ", len(train_title))
    print("len of train_text is ", len(train_text))

    # Print shapes to verify
    print("Training set:")
    print("Title shape:", train_title.shape)
    print("Text shape:", train_text.shape)
    print("Labels shape:", y_train.shape)

    print("\nTest set:")
    print("Title shape:", test_title.shape)
    print("Text shape:", test_text.shape)
    print("Labels shape:", y_test.shape)
    x_train, x_test = np.c_[train_title, train_text], np.c_[test_title, test_text]
    print(f"x_train shape is {x_train.shape}, y_train shape is {y_train.shape}")
    return x_train, x_test, y_train, y_test

def cross_validate_svm(x_train, y_train, kernel):
    """
    5-fold cross-validation on (x_train, y_train) using an SVM pipeline
    that includes scaling. Prints mean accuracy across folds.
    """
    # Build a pipeline: scale -> SVM
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, random_state=42))
    ])

    # StratifiedKFold for classification tasks
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate cross-validation accuracy
    scores = cross_val_score(pipeline, x_train, y_train, cv=kf, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {scores}")
    print(f"Mean CV accuracy: {scores.mean():.4f}  (std: {scores.std():.4f})")

def train_and_evaluate(x_train, x_test, y_train, y_test, kernel):

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    svm_linear = SVC(kernel=kernel, random_state=42)
    svm_linear.fit(x_train_scaled, y_train)

    y_pred_svm = svm_linear.predict(x_test_scaled)
    #print("Unique predictions:", np.unique(y_pred_svm))

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_svm)
    cm = confusion_matrix(y_test, y_pred_svm)
    precision = precision_score(y_test, y_pred_svm)
    recall = recall_score(y_test, y_pred_svm)  # Recall is also sensitivity
    f1 = f1_score(y_test, y_pred_svm)

    # Compute specificity: TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # Print the results
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)
    print("Precision:", precision)
    print("Recall (Sensitivity):", recall)
    print("F1 Score:", f1)
    print("Specificity:", specificity)

if __name__ == "__main__":
    data_npz = np.load(os.path.join(os.path.dirname(__file__), "../", "src", "data", "preprocessed_data_svm.npz"))
    x_train, x_test, y_train, y_test = split_train_test(data_npz)
    # Q5 - Training the data using simple ML algorithm - SVM with kernel = rbf
    kernel = "rbf"
    cross_validate_svm(x_train, y_train, kernel=kernel)
    train_and_evaluate(x_train, x_test, y_train, y_test, kernel)
