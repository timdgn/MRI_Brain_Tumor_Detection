from warnings import filterwarnings
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from constants import *
import preprocessing
import datetime
import os
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score


def create_model():
    """
    Create EfficientNet model with specified input shape and output size.

    Returns:
        model (object): The trained machine learning model.
    """

    # Create EfficientNetB0 model with pre-trained weights
    effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    model = tf.keras.layers.GlobalAveragePooling2D()(effnet.output)
    model = tf.keras.layers.Dropout(rate=0.5)(model)
    model = tf.keras.layers.Dense(4, activation='softmax')(model)
    model = tf.keras.models.Model(inputs=effnet.input, outputs=model)

    return model


def train_model(model, X, y, now):
    """
    Train a machine learning model using the specified data and training parameters.

    Parameters:
        model (object): The machine learning model to be trained.
        X (array-like): The input data for training the model.
        y (array-like): The target data for training the model.
        now (datetime): The current date and time.

    Returns:
        history (object): The history object containing information about the training process.
    """

    print('Starting training...')

    # Compile the model with the specified loss, optimizer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Set up callbacks for TensorBoard, model checkpointing, and learning rate reduction
    output_dir = os.path.join(os.getcwd(), '..', 'models')
    filename = f"effnet_{now.strftime('%Y-%m-%d_%H-%M-%S')}.keras"
    checkpoint = ModelCheckpoint(os.path.join(output_dir, filename), monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto',
                                  verbose=1)

    # Train the model with the specified data and training parameters
    history = model.fit(X, y, validation_split=0.1, epochs=EPOCHS, verbose=1, batch_size=32,
                        callbacks=[checkpoint, reduce_lr])

    print('Training done...', end='\n\n')

    return history


def plot_history(history, now):
    """
    Generate a plot to visualize the training and validation accuracy/loss over epochs.

    Parameters:
        history: A dictionary containing the training and validation accuracy/loss history.
        now: A datetime object representing the current date and time.

    Returns:
        None
    """
    filterwarnings('ignore')

    epochs = [i+1 for i in range(EPOCHS)]
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    # fig.text(s='Epochs vs. Training and Validation Accuracy/Loss', size=18, fontweight='bold',
    #          fontname='monospace', color=COLORS_DARK[1], y=1, x=0.28, alpha=0.8)
    fig.suptitle('Epochs vs. Training and Validation Accuracy/Loss',
                 fontsize=18, fontweight='bold', color=COLORS_DARK[1])

    sns.despine()
    ax[0].plot(epochs, train_acc, marker='o', markerfacecolor=COLORS_GREEN[2], color=COLORS_GREEN[3],
               label='Training Accuracy')
    ax[0].plot(epochs, val_acc, marker='o', markerfacecolor=COLORS_RED[2], color=COLORS_RED[3],
               label='Validation Accuracy')
    ax[0].legend(frameon=False)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')

    sns.despine()
    ax[1].plot(epochs, train_loss, marker='o', markerfacecolor=COLORS_GREEN[2], color=COLORS_GREEN[3],
               label='Training Loss')
    ax[1].plot(epochs, val_loss, marker='o', markerfacecolor=COLORS_RED[2], color=COLORS_RED[3],
               label='Validation Loss')
    ax[1].legend(frameon=False)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')

    output_dir = os.path.join(os.getcwd(), '..', 'plots', 'history')
    filename = f"Accuracy_Loss_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

    fig.show()


def load_last_model():
    """
    Loads the last model created from the specified folder.

    Returns:
        model: The loaded model.
    """

    models_dir = os.path.join(os.getcwd(), '..', 'models')
    files = os.listdir(models_dir)
    filename = sorted(files)[-1]

    model = tf.keras.models.load_model(os.path.join(models_dir, filename))

    return model


def predict(model, X_test, y_test):
    """
    Generate the predicted labels for the given test data using the provided model.

    Parameters:
        model (object): The trained model object used for prediction.
        X_test (array-like): The test data to be used for prediction.
        y_test (array-like): The ground truth labels for the test data.

    Returns:
        tuple: A tuple containing the predicted labels and the ground truth labels as arrays.
    """

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    return y_pred, y_test


def plot_conf_matrix(y_pred, y_test, now):
    """
    Generates a heatmap of the confusion matrix based on the predicted labels and the actual labels.

    Parameters:
        y_pred (array-like): An array or a list-like object containing the predicted labels.
        y_test (array-like): An array or a list-like object containing the actual labels.
        now (datetime.datetime): A datetime object representing the current date and time.

    Returns:
        None
    """

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), ax=ax, xticklabels=LABELS, yticklabels=LABELS, annot=True,
                cmap=COLORS_GREEN[::-1], alpha=0.7, linewidths=2, linecolor=COLORS_DARK[3])
    ax.set_title('Heatmap of the Confusion Matrix', size=18, fontweight='bold',
                color=COLORS_DARK[1])

    output_dir = os.path.join(os.getcwd(), '..', 'plots', 'confusion')
    filename = f"Confusion_Matrix_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

    plt.show()


def plot_metrics(y_pred, y_test, now):
    """
    Generates a lollipop plot of precision, F1 score, and recall based on the predicted labels and the actual labels.

    Parameters:
        y_pred (array-like): An array or a list-like object containing the predicted labels.
        y_test (array-like): An array or a list-like object containing the actual labels.
        now (datetime.datetime): A datetime object representing the current date and time.
    Returns:
        None
    """

    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    metrics = {'Precision': precision, 'F1 Score': f1, 'Recall': recall}

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.vlines(x=list(metrics.keys()), ymin=0, ymax=list(metrics.values()), color=COLORS_GREEN[::-1], alpha=0.7)
    ax.plot(list(metrics.keys()), list(metrics.values()), "o", color=COLORS_GREEN[0])
    ax.set_title('Metrics', size=18, fontweight='bold', color=COLORS_DARK[1])
    ax.grid(axis='y')

    output_dir = os.path.join(os.getcwd(), '..', 'plots', 'metrics')
    filename = f"Metrics_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

    plt.show()


def main(X_train, X_test, y_train, y_test):
    """
    Trains a model with the provided training data, saves the trained model,
    and generates predictions and a confusion matrix.

    Parameters:
        X_train (numpy.ndarray): Training data features.
        X_test (numpy.ndarray): Test data features.
        y_train (numpy.ndarray): Training data labels.
        y_test (numpy.ndarray): Test data labels.
    """
    now = datetime.datetime.now()

    model = create_model()
    history = train_model(model, X_train, y_train, now)
    plot_history(history, now)

    model = load_last_model()
    y_pred, y_test = predict(model, X_test, y_test)
    plot_conf_matrix(y_pred, y_test, now)
    plot_metrics(y_pred, y_test, now)


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = preprocessing.main()

    main(X_train, X_test, y_train, y_test)
