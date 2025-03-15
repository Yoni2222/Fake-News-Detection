import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from src.metrics import MetricsCallback


results_dir = '../results'


def split_train_test(preprocessed_data, removed_data=None, imbalance_level=None):


    # Retrieve arrays from the NPZ file
    text_clean, title_clean, labels = preprocessed_data["text_clean"], preprocessed_data["title_clean"], \
                                          preprocessed_data["labels"]
    print("len of labels is ", len(labels))
    N = len(labels)
    cutoff = int(0.9 * N)
    texts, titles, labels_cut = text_clean[:cutoff], title_clean[:cutoff], labels[:cutoff]

    train_title, test_title, train_text, test_text, y_train, y_test = train_test_split(
        titles, texts, labels_cut, test_size=0.2, random_state=42
    )

    # Q10-11, take only 60% of the train_data to test the model accuracy when using less data samples for training
    if removed_data == True:
        cutoff = int(0.6 * N)
        train_text, train_title, y_train = train_text[:cutoff], train_title[:cutoff], y_train[:cutoff]
    # Q10-11, increasing training set num of samples to test the model accuracy when using more data samples for training
    elif removed_data == False:
        train_text = np.concatenate([train_text, text_clean[cutoff:]], axis=0)
        train_title = np.concatenate([train_title, title_clean[cutoff:]], axis=0)
        y_train = np.concatenate([y_train, labels[cutoff:]], axis=0)

    # Q14 - Create different imbalance levels
    if imbalance_level is not None:
        # Separate samples by class
        indices_class_0 = np.where(y_train == 0)[0]
        indices_class_1 = np.where(y_train == 1)[0]

        total_samples = len(y_train)

        if imbalance_level == 0:
            # 0% class 1 (extreme imbalance - only class 0)
            selected_indices = indices_class_0

        elif imbalance_level == 35:
            # Exactly 35% class 1
            target_class_1_count = int(0.35 * total_samples)
            target_class_0_count = total_samples - target_class_1_count

            # Handle case where we don't have enough samples
            if len(indices_class_1) < target_class_1_count:
                # Use all class 1 with replacement
                selected_class_1 = np.random.choice(indices_class_1, target_class_1_count, replace=True)
            else:
                # Use random selection without replacement
                selected_class_1 = np.random.choice(indices_class_1, target_class_1_count, replace=False)

            if len(indices_class_0) < target_class_0_count:
                # Use all class 0 with replacement
                selected_class_0 = np.random.choice(indices_class_0, target_class_0_count, replace=True)
            else:
                # Use random selection without replacement
                selected_class_0 = np.random.choice(indices_class_0, target_class_0_count, replace=False)

            # Combine and shuffle
            selected_indices = np.concatenate([selected_class_0, selected_class_1])
            np.random.shuffle(selected_indices)

        elif imbalance_level == 80:
            # Exactly 80% class 1
            target_class_1_count = int(0.8 * total_samples)
            target_class_0_count = total_samples - target_class_1_count

            # Handle case where we don't have enough samples
            if len(indices_class_1) < target_class_1_count:
                # Use all class 1 with replacement
                selected_class_1 = np.random.choice(indices_class_1, target_class_1_count, replace=True)
            else:
                # Use random selection without replacement
                selected_class_1 = np.random.choice(indices_class_1, target_class_1_count, replace=False)

            if len(indices_class_0) < target_class_0_count:
                # Use all class 0 with replacement
                selected_class_0 = np.random.choice(indices_class_0, target_class_0_count, replace=True)
            else:
                # Use random selection without replacement
                selected_class_0 = np.random.choice(indices_class_0, target_class_0_count, replace=False)

            # Combine and shuffle
            selected_indices = np.concatenate([selected_class_0, selected_class_1])
            np.random.shuffle(selected_indices)

            # Combine the selected indices
            selected_indices = np.concatenate([selected_class_0, selected_class_1])

        # Apply the selected indices to create the imbalanced dataset
        train_title = train_title[selected_indices]
        train_text = train_text[selected_indices]
        y_train = y_train[selected_indices]
    numOfSamples = len(y_train)
    numOfTrueSamples = np.sum(y_train)
    numOfFalseSamples = numOfSamples - numOfTrueSamples

    print("len of labels is ", numOfSamples)
    print("num of 1 labels is ", numOfTrueSamples)
    print("num of 0 labels is ", numOfFalseSamples)
    print(f"{numOfTrueSamples / numOfSamples:.2f}% is Fake, {numOfFalseSamples / numOfSamples:.2f}% is Real")


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
    # x_train, x_test = train_text, test_text
    print(f"x_train shape is {x_train.shape}, y_train shape is {y_train.shape}")
    return x_train, x_test, y_train, y_test


def train_and_evaluate(x_train, x_test, y_train, y_test, epochs=1, optimizer="RMSprop", batch=32, removed_data=None,
                       improved_arch=False, reduced_dim=False):

    if reduced_dim:
        vocab_size = 5000
        embedding_dim = 32
        timesteps = 256
    elif improved_arch:
        vocab_size = 13000
        embedding_dim = 84  # number of features for each token
        timesteps = 512
    else:
        vocab_size = 20000
        embedding_dim = 128  # number of features for each token
        timesteps = 512

    units = 32  # hidden dimension of the LSTM

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                        input_length=timesteps))  # (batch_size, 512) -> (batch_size, 512, 128)



    if improved_arch:
        model.add(SpatialDropout1D(0.2))  # Dropout for embeddings
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.0))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))  # Add L2 regularization

        # early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Learning rate scheduler
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    else:
        model.add(LSTM(units=units))
        model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    modelSummary = model.summary()
    print(modelSummary)

    x_train = np.clip(x_train, 0, vocab_size - 1)  # 13000-1
    x_test = np.clip(x_test, 0, vocab_size - 1)

    # Create validation split
    train_idx = int(0.8 * len(x_train))
    x_train_final, x_val = x_train[:train_idx], x_train[train_idx:]
    y_train_final, y_val = y_train[:train_idx], y_train[train_idx:]

    # Initialize our custom metrics callback
    metrics_callback = MetricsCallback(training_data=(x_train_final, y_train_final), validation_data=(x_val, y_val))

    callbacks = [metrics_callback]
    if improved_arch:
        callbacks.extend([early_stopping, reduce_lr])

    history = model.fit(x_train_final, y_train_final, epochs=epochs, batch_size=batch, validation_data=(x_val, y_val), callbacks=callbacks, verbose=1)

    # Plot training vs. validation loss
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    axs[0, 0].plot(history.history['loss'], label='train_loss')
    axs[0, 0].plot(history.history['val_loss'], label='val_loss')
    axs[0, 0].set_title('Loss over epochs')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # MCC plot with both training and validation
    axs[0, 1].plot(metrics_callback.train_mcc_scores, label='Training MCC')
    axs[0, 1].plot(metrics_callback.val_mcc_scores, label='Validation MCC')
    axs[0, 1].set_title('Matthews Correlation Coefficient')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('MCC')
    axs[0, 1].legend()

    # AUC plot with both training and validation
    axs[1, 0].plot(metrics_callback.train_auc_scores, label='Training AUC')
    axs[1, 0].plot(metrics_callback.val_auc_scores, label='Validation AUC')
    axs[1, 0].set_title('Area Under ROC Curve')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('AUC')
    axs[1, 0].legend()

    # ROC curve for the last epoch (training, validation, and test)
    # Get the last epoch's ROC curves
    if len(metrics_callback.train_fpr_history) > 0:
        last_train_fpr = metrics_callback.train_fpr_history[-1]
        last_train_tpr = metrics_callback.train_tpr_history[-1]
        last_train_auc = metrics_callback.train_auc_scores[-1]

        last_val_fpr = metrics_callback.val_fpr_history[-1]
        last_val_tpr = metrics_callback.val_tpr_history[-1]
        last_val_auc = metrics_callback.val_auc_scores[-1]

        # Test set ROC
        y_pred_proba = model.predict(x_test)
        test_fpr, test_tpr, _ = roc_curve(y_test, y_pred_proba)
        test_auc = auc(test_fpr, test_tpr)

        # Plot all three ROC curves
        axs[1, 1].plot(last_train_fpr, last_train_tpr, label=f'Training (AUC={last_train_auc:.3f})')
        axs[1, 1].plot(last_val_fpr, last_val_tpr, label=f'Validation (AUC={last_val_auc:.3f})')
        axs[1, 1].plot(test_fpr, test_tpr, label=f'Test (AUC={test_auc:.3f})')
        axs[1, 1].plot([0, 1], [0, 1], 'k--')  # Diagonal line
        axs[1, 1].set_xlim([0.0, 1.0])
        axs[1, 1].set_ylim([0.0, 1.05])
        axs[1, 1].set_xlabel('False Positive Rate')
        axs[1, 1].set_ylabel('True Positive Rate')
        axs[1, 1].set_title('ROC Curves (Last Epoch)')
        axs[1, 1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # 2) Save the plot to the results directory
    file_name = f"{epochs} epochs_batch:{batch}_optimizer:{optimizer}"
    plot_path = os.path.join(results_dir, f"{file_name}.png")
    fig.savefig(plot_path)
    plt.close(fig)  # Close figure so it doesn't pop up repeatedly in the notebook

    model_save_path = os.path.join(results_dir, f"model_{file_name}.h5")
    model.save(model_save_path)
    print("Trained model saved to:", model_save_path)

    # evaluate the test set
    y_pred = (model.predict(x_test) > 0.5).astype(int).flatten()

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics_path = os.path.join(results_dir, f"metrics of {file_name}.txt")
    with open(metrics_path, 'w') as f_out:
        f_out.write(f"=== Results (epochs={epochs}) ===\n")
        f_out.write(f"Accuracy: {acc}\n")
        f_out.write(f"Confusion Matrix:\n{cm}\n")
        f_out.write(f"Precision: {prec}\n")
        f_out.write(f"Recall (Sensitivity): {rec}\n")
        f_out.write(f"F1 Score: {f1}\n")
        f_out.write(f"Specificity: {specificity}\n")
        f_out.write(f"Model summary: {modelSummary}\n")
        f_out.write(f"Matthews Correlation Coefficient: {mcc}\n")
        f_out.write(f"AUC: {test_auc}\n")

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Precision:", prec)
    print("Recall (Sensitivity):", rec)
    print("F1 Score:", f1)
    print("Specificity:", specificity)
    print("Matthews Correlation Coefficient:", mcc)
    print("AUC:", test_auc)

if __name__ == "__main__":
    # loading preprocessed data
    data_npz = np.load(os.path.join(os.path.dirname(__file__), "..", "src", "data", "preprocessed_data.npz"))

    # Q10-11 - splitting the data to train, val and test using 60% of the data and 100% of the data
    #x_train, x_test, y_train, y_test = split_train_test(data_npz, removed_data = True)
    #x_train, x_test, y_train, y_test = split_train_test(data_npz, removed_data = False)


    # Q14 - changing data imbalance to 3 different levels:
    #   1.)  0% - class 1, 100% - class 0.
    #   2.)  35% - class 1, 65% - class 0.
    #   3.)  80% - class 1, 20% - class 0.
    #x_train, x_test, y_train, y_test = split_train_test(data_npz, imbalance_level=0)
    # x_train, x_test, y_train, y_test = split_train_test(data_npz, imbalance_level=35)
    # x_train, x_test, y_train, y_test = split_train_test(data_npz, imbalance_level=80)

    x_train, x_test, y_train, y_test = split_train_test(data_npz)

    ###################################Q7 - Training a model with default hyperparameters: 1 epoch, batch size of 32, optimizer = "RMSprop"###################################
    train_and_evaluate(x_train, x_test, y_train, y_test)


    ####################################Q9 - Trying 3 different hyperparameters and training a network with 3 different values for each hyperparameter. ###################################
    # The chosen hyperparameters are epoch numbers, batch size and optimizer
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 10)
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 50)
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 100)
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 200)

    # epochs = 50 is not a default value but there are no major differences between the results when training for 1 epoch
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 50, optimizer="Adam")
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 50, optimizer="Adagrad")
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 50, optimizer="SGD")

    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 50, batch=16)
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 50, batch=64)
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 50, batch=128)


    #Q12 - Training a model with better architecture -> lower vocab_size, lower embedding_dim, SpatialDropout1D(0.2) after embedding layer, adding drouput(0.2) to LSTM layer,
    # adding L2 regularizer to Dense layer, adding Early Stopping and adding ReduceLROnPlateau. Spoiler: This model gave the best results
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs=100, batch=16, improved_arch=True)

    #Q15 - Training a model with reduced dimensions -> -> lower vocab_size(5000), lower embedding_dim(64), lower timesteps(256)
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs=100, batch=16, improved_arch=True, reduced_dim=True)



    #train_and_evaluate(x_train, x_test, y_train, y_test)
    #train_and_evaluate(x_train, x_test, y_train, y_test, epochs = 100, batch = 16, reduced = False)

