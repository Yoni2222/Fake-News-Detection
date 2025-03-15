import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_curve, auc
from tensorflow.keras.callbacks import Callback

class MetricsCallback(Callback):
    def __init__(self, training_data, validation_data):
        super(MetricsCallback, self).__init__()
        self.x_train, self.y_train = training_data
        self.x_val, self.y_val = validation_data

        # For validation data
        self.val_mcc_scores = []
        self.val_auc_scores = []
        self.val_fpr_history = []
        self.val_tpr_history = []

        # For training data
        self.train_mcc_scores = []
        self.train_auc_scores = []
        self.train_fpr_history = []
        self.train_tpr_history = []

    def on_epoch_end(self, epoch, logs=None):
        # Calculate metrics on validation data
        val_pred_proba = self.model.predict(self.x_val)
        val_pred = (val_pred_proba > 0.5).astype(int).flatten()

        val_mcc = matthews_corrcoef(self.y_val, val_pred)
        self.val_mcc_scores.append(val_mcc)

        val_fpr, val_tpr, _ = roc_curve(self.y_val, val_pred_proba)
        val_roc_auc = auc(val_fpr, val_tpr)

        self.val_fpr_history.append(val_fpr)
        self.val_tpr_history.append(val_tpr)
        self.val_auc_scores.append(val_roc_auc)

        # Calculate metrics on training data (using a subset to save computation)
        # Using a random sample of 20% of training data to speed up computation
        indices = np.random.choice(len(self.x_train), size=min(1000, len(self.x_train)), replace=False)
        x_train_sample = self.x_train[indices]
        y_train_sample = self.y_train[indices]

        train_pred_proba = self.model.predict(x_train_sample)
        train_pred = (train_pred_proba > 0.5).astype(int).flatten()

        train_mcc = matthews_corrcoef(y_train_sample, train_pred)
        self.train_mcc_scores.append(train_mcc)

        train_fpr, train_tpr, _ = roc_curve(y_train_sample, train_pred_proba)
        train_roc_auc = auc(train_fpr, train_tpr)

        self.train_fpr_history.append(train_fpr)
        self.train_tpr_history.append(train_tpr)
        self.train_auc_scores.append(train_roc_auc)

        # Print metrics for this epoch
        print(
            f"\nEpoch {epoch + 1} - Train: MCC={train_mcc:.4f}, AUC={train_roc_auc:.4f} | Val: MCC={val_mcc:.4f}, AUC={val_roc_auc:.4f}")