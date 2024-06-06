import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from  sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score
import random
import time
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from tensorflow.python.layers.core import Dense
#from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
import tensorflow_addons as tfa
from dataloader import SeqDataLoader
import argparse
from tensorboard.plugins.hparams import api as hp
from typing import Any
from dataclasses import dataclass
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, TimeDistributed, Attention
from tensorflow.keras import Model
from tensorflow.keras import layers 
from sklearn.decomposition import FastICA
from io import StringIO



@dataclass
class HParams:
    epochs: int = 120  # 120, 300 ###################################change
    batch_size: int = 40  # 20, 10
    num_units: int = 128
    embed_size: int = 10
    input_depth: int = 3000
    #n_channels: int = 40  ###100e
    max_time_step: int = 40  # 5 3 second best 10# 40 # 100
    output_max_length: int = 11  # max_time_step +1
    akara2017: bool = True
    test_step: int = 5  # each 10 epochs
    num_folds: int= 5  ######################################### change
    num_classes: int = 5
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, HParams) and self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))
tf.compat.v1.disable_v2_behavior()

def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
#     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size
def flatten(name, input_var):
    dim = 1
    for d in input_var.get_shape()[1:].as_list():
        dim *= d
    output_var = tf.reshape(input_var,
                            shape=[-1, dim],
                            name=name)

    return output_var
class FLAGS:
    num_folds = HParams.num_folds
    data_dir = 'data_2013/eeg_fpz_cz'
    output_dir = 'outputs_2013/outputs_eeg_fpz_cz'
    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    checkpoint_dir = 'checkpoints-seq2seq-sleep-EDF'

def build_combined_model(input_shape, hparams, keep_prob=0.5):
    inputs = tf.keras.Input(shape=input_shape)
    
    # First part of the model
    x1 = layers.Conv1D(64, 50, strides=6, padding='same', activation='relu')(inputs)
    x1 = layers.MaxPooling1D(8, strides=8, padding='same')(x1)
    x1 = layers.Dropout(keep_prob)(x1)
    x1 = layers.Conv1D(128, 8, strides=1, padding='same', activation='relu')(x1)
    x1 = layers.Conv1D(128, 8, strides=1, padding='same', activation='relu')(x1)
    x1 = layers.Conv1D(128, 8, strides=1, padding='same', activation='relu')(x1)
    x1 = layers.MaxPooling1D(4, strides=4, padding='same')(x1)
    x1 = layers.Conv1D(256, 8, strides=1, padding='same', activation='relu')(x1)
    x1 = layers.MaxPooling1D(4, strides=4, padding='same')(x1)
    x1 = layers.Flatten()(x1)

    # Second part of the model
    x2 = layers.Conv1D(64, 400, strides=50, padding='same', activation='relu')(inputs)
    x2 = layers.MaxPooling1D(4, strides=4, padding='same')(x2)
    x2 = layers.Dropout(keep_prob)(x2)
    x2 = layers.Conv1D(128, 6, strides=1, padding='same', activation='relu')(x2)
    x2 = layers.Conv1D(128, 6, strides=1, padding='same', activation='relu')(x2)
    x2 = layers.Conv1D(128, 6, strides=1, padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling1D(2, strides=2, padding='same')(x2)
    x2 = layers.Conv1D(256, 6, strides=1, padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling1D(2, strides=2, padding='same')(x2)
    x2 = layers.Flatten()(x2)

    # Combine parts
    concatenated = layers.concatenate([x1, x2])
    concatenated = layers.Dropout(keep_prob)(concatenated)
    
    # Additional layers
    x = layers.Reshape((1, concatenated.shape[1]))(concatenated)  # Reshape to add a dummy dimension
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(keep_prob)(x)
    x = layers.Conv1D(hparams.num_classes, 1, padding='same', activation='softmax')(x)  # Final layer for classification
    x = layers.Flatten()(x)  # Flatten to ensure the output shape matches the expected shape
    
    # Create the full model
    model = tf.keras.Model(inputs, x)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])  # Use sparse_categorical_accuracy for sparse labels
    return model


class LossAccuracyTracker(tf.keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs["loss"])
        self.accuracies.append(logs["sparse_categorical_accuracy"])
        self.val_losses.append(logs["val_loss"])
        self.val_accuracies.append(logs["val_sparse_categorical_accuracy"])

def plot_metrics(loss_acc_tracker, file_prefix):
    epochs = range(1, len(loss_acc_tracker.losses) + 1)

    plt.figure(figsize=(12, 8))

    # Plot Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_acc_tracker.losses, label='Training Loss')
    plt.plot(epochs, loss_acc_tracker.val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(ticks=epochs)  # Ensure x-axis labels are integers

    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss_acc_tracker.accuracies, label='Training Accuracy')
    plt.plot(epochs, loss_acc_tracker.val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(ticks=epochs)  # Ensure x-axis labels are integers

    plt.tight_layout()
    plt.savefig(f'{file_prefix}_metrics.png')
    plt.show()

def train_model(model, train_data, train_labels, val_data, val_labels, epochs=10):
    train_labels = train_labels.reshape(-1)
    val_labels = val_labels.reshape(-1)
    train_data = train_data.reshape(-1, 3000, 1)
    val_data = val_data.reshape(-1, 3000, 1)
    print(train_data.shape)
    print(val_data.shape)
    loss_acc_tracker = LossAccuracyTracker()

    history = model.fit(train_data, train_labels, epochs=epochs,
                        validation_data=(val_data, val_labels),
                        callbacks=[loss_acc_tracker])
    return history, loss_acc_tracker

def evaluate_model(model, test_data, test_labels, num_classes, file_path,loss_acc_tracker):
    eval_results = model.evaluate(test_data, test_labels)
    print(f"Test loss: {eval_results[0]}, Test accuracy: {eval_results[1]}")

    y_pred = np.argmax(model.predict(test_data), axis=-1)
    y_true = test_labels

    cm = confusion_matrix(y_true, y_pred)
    ck_score = cohen_kappa_score(y_true, y_pred)
    acc_avg, acc, f1_macro, f1, sensitivity, specificity, PPV = evaluate_metrics(cm, num_classes)
    
    with open(file_path, 'a') as f:
        f.write(f"Test loss: {eval_results[0]}, Test accuracy: {eval_results[1]}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Cohen's Kappa: {ck_score}\n")
        f.write(f"Average Accuracy: {acc_avg}, Macro F1: {f1_macro}\n")
        
        for i in range(num_classes):
            f.write(f"Class {i}: Sensitivity: {sensitivity[i]}, Specificity: {specificity[i]}, Precision: {PPV[i]}, F1: {f1[i]}, Accuracy: {acc[i]}\n")
        
        f.write(f"Overall - Sensitivity: {np.mean(sensitivity)}, Specificity: {np.mean(specificity)}, Precision: {np.mean(PPV)}, F1-score: {np.mean(f1)}, Accuracy: {np.mean(acc)}\n")
        
        f.write("\nEpoch-wise Loss and Accuracy:\n")
        for epoch in range(len(loss_acc_tracker.losses)):
            f.write(f"Epoch {epoch + 1}: Loss: {loss_acc_tracker.losses[epoch]}, "
                    f"Accuracy: {loss_acc_tracker.accuracies[epoch]}, "
                    f"Val Loss: {loss_acc_tracker.val_losses[epoch]}, "
                    f"Val Accuracy: {loss_acc_tracker.val_accuracies[epoch]}\n")

    print(f"Confusion Matrix:\n{cm}")
    print(f"Cohen's Kappa: {ck_score}")
    print(f"Average Accuracy: {acc_avg}, Macro F1: {f1_macro}")
    
    for i in range(num_classes):
        print(f"Class {i}: Sensitivity: {sensitivity[i]}, Specificity: {specificity[i]}, Precision: {PPV[i]}, F1: {f1[i]}, Accuracy: {acc[i]}")
    
    print(f"Overall - Sensitivity: {np.mean(sensitivity)}, Specificity: {np.mean(specificity)}, Precision: {np.mean(PPV)}, F1-score: {np.mean(f1)}, Accuracy: {np.mean(acc)}")

def evaluate_metrics(cm, num_classes):
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    FDR = FP / (TP + FP)

    ACC = (TP + TN) / (TP + FP + FN + TN)
    ACC_macro = np.mean(ACC)
    F1 = (2 * PPV * TPR) / (PPV + TPR)
    F1_macro = np.mean(F1)

    return ACC_macro, ACC, F1_macro, F1, TPR, TNR, PPV

# def load_data(data_dir, num_folds, fold_idx, classes, max_time_step):
#     # Implement your data loading logic here
#     # For now, we'll mock the data
#     # Replace this with actual data loading and preprocessing
#     X_train = np.random.rand(100, max_time_step, 1)
#     y_train = np.random.randint(0, len(classes), 100)
#     X_test = np.random.rand(20, max_time_step, 1)
#     y_test = np.random.randint(0, len(classes), 20)
#     return X_train, y_train, X_test, y_test


def run_program(hparams, FLAGS):
    num_folds = FLAGS.num_folds
    data_dir = FLAGS.data_dir
    output_dir = FLAGS.output_dir
    classes = FLAGS.classes
    n_classes = len(classes)

    for fold_idx in range(num_folds):
        start_time_fold_i = time.time()
        data_loader = SeqDataLoader(data_dir, num_folds, fold_idx, classes=classes)
        X_train, y_train, X_test, y_test = data_loader.load_data(seq_len=hparams.max_time_step)
        ################################# ICA ##########################################  
        # n_samples, n_channels, n_timepoints = X_train.shape

        # # Initialize arrays to store the transformed data
        # X_train_ica = np.zeros_like(X_train)
        # X_test_ica = np.zeros_like(X_test)

        # # Define the number of components for ICA
        # n_components = min(n_timepoints, n_samples)  # Choose a feasible number of components

        # # Apply ICA separately on each channel
        # for i in range(n_channels):
        #     # Extract the data for the current channel across all samples
        #     X_train_channel = X_train[:, i, :]  # Shape: (n_samples, n_timepoints)
        #     X_test_channel = X_test[:, i, :]    # Shape: (number of test samples, n_timepoints)
            
        #     # Apply ICA on the current channel data
        #     ica = FastICA(n_components=n_components, random_state=0)
        #     X_train_ica_channel = ica.fit_transform(X_train_channel)  # Shape: (n_samples, n_components)
        #     X_test_ica_channel = ica.transform(X_test_channel)        # Shape: (number of test samples, n_components)
            
        #     # Pad the transformed data to match the original number of timepoints
        #     if X_train_ica_channel.shape[1] < n_timepoints:
        #         # Pad with zeros
        #         X_train_ica_channel = np.pad(X_train_ica_channel, ((0, 0), (0, n_timepoints - n_components)), mode='constant')
        #         X_test_ica_channel = np.pad(X_test_ica_channel, ((0, 0), (0, n_timepoints - n_components)), mode='constant')
            
        #     # Assign the transformed data back to the respective place
        #     X_train_ica[:, i, :] = X_train_ica_channel
        #     X_test_ica[:, i, :] = X_test_ica_channel

        # # Verify shapes
        # print(X_train_ica.shape)  # Should be (n_samples, n_channels, n_timepoints)
        # print(X_test_ica.shape)
        # X_train = X_train_ica
        # X_test = X_test_ica
        ################################# ICA ##########################################  
        # Flatten the data to treat each sub-sequence as an individual sample
        X_train = X_train.reshape(-1, 3000, 1)
        y_train = y_train.reshape(-1)
        X_test = X_test.reshape(-1, 3000, 1)
        y_test = y_test.reshape(-1)

        # Shuffle the training data
        permute = np.random.permutation(len(y_train))
        X_train = X_train[permute]
        y_train = y_train[permute]

    input_shape = (3000, 1)  # Set input shape to match the flattened data
    print(X_train.shape)  # Should print (840*40, 3000, 1)
    print(y_train.shape)  # Should print (840*40,)
    
    model = build_combined_model(input_shape, hparams)

    model_summary = StringIO()
    model.summary(print_fn=lambda x: model_summary.write(x + '\n'))
    summary_str = model_summary.getvalue()

    # Write the model summary to the evaluation results file
    with open('evaluation_results.txt', 'w') as f:
        f.write("Model Summary:\n")
        f.write(summary_str)
        f.write("\n")  # Add an extra line for better readability

    
    history, loss_acc_tracker = train_model(model, X_train, y_train, X_test, y_test, epochs=hparams.epochs)
    evaluate_model(model, X_test, y_test, n_classes, 'evaluation_results.txt',loss_acc_tracker)
    plot_metrics(loss_acc_tracker, 'evaluation_results')


def main(args=None):
    hparams = HParams()
    run_program(hparams, FLAGS)

if __name__ == "__main__":
    main()