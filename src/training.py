import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from settings import *
import preprocessing


def create_model():
    """
    Create EfficientNet model with specified input shape and output size.

    Returns:
        model (object): The trained machine learning model.
    """

    # Create EfficientNetB0 model with pre-trained weights
    effnet = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False,
                                                  input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    model = tf.keras.layers.GlobalAveragePooling2D()(effnet.output)
    model = tf.keras.layers.Dropout(rate=0.5)(model)
    model = tf.keras.layers.Dense(4, activation='softmax')(model)
    model = tf.keras.models.Model(inputs=effnet.input, outputs=model)

    return model


def train_model(model, X, y):
    """
    Train a machine learning model using the specified data and training parameters.

    Parameters:
        model (object): The machine learning model to be trained.
        X (array-like): The input data for training the model.
        y (array-like): The target data for training the model.

    Returns:
        history (object): The history object containing information about the training process.
    """

    print('Starting training...')

    # Compile the model with the specified loss, optimizer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Set up callbacks for TensorBoard, model checkpointing, and learning rate reduction
    now = datetime.datetime.now()
    output_dir = os.path.join(PROJECT_DIR, 'models')
    filename = f"effnet_{now.strftime('%Y-%m-%d_%H-%M-%S')}.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, filename), monitor="val_accuracy",
                                                    save_best_only=True, mode="auto", verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001,
                                                     mode='auto',
                                                     verbose=1)

    # Train the model with the specified data and training parameters
    history = model.fit(X, y, validation_split=0.1, epochs=EPOCHS, verbose=1, batch_size=32,
                        callbacks=[checkpoint, reduce_lr])

    print('Training done...', end='\n\n')

    return model, history


def plot_history(history):
    """
    Generate a plot to visualize the training and validation accuracy/loss over epochs and save the plots

    Parameters:
        history: A dictionary containing the training and validation accuracy/loss history.

    Returns:
        None
    """

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history.history['accuracy']) + 1)  # Adjust epochs to start from 1
    plt.plot(epochs, history.history['accuracy'])
    plt.plot(epochs, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Annotate the highest point for accuracy
    max_acc = max(history.history['accuracy'])
    max_val_acc = max(history.history['val_accuracy'])
    plt.annotate(f'Max Train Accuracy: {format(max_acc, ".3f")}',
                 xy=(history.history['accuracy'].index(max_acc) + 1, max_acc), xytext=(10, -20),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'Max Validation Accuracy: {format(max_val_acc, ".3f")}',
                 xy=(history.history['val_accuracy'].index(max_val_acc) + 1, max_val_acc), xytext=(10, -20),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'))

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'])
    plt.plot(epochs, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Annotate the lowest point for loss
    min_loss = min(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    plt.annotate(f'Min Train Loss: {format(min_loss, ".3f")}',
                 xy=(history.history['loss'].index(min_loss) + 1, min_loss), xytext=(10, 20),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    plt.annotate(f'Min Validation Loss: {format(min_val_loss, ".3f")}',
                 xy=(history.history['val_loss'].index(min_val_loss) + 1, min_val_loss), xytext=(10, 20),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()

    output_dir = os.path.join(PROJECT_DIR, 'plots', 'history')
    filename = f"Accuracy_Loss.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

    plt.show()


def load_last_model():
    """
    Loads the last model created from the specified folder.

    Returns:
        model: The loaded model.
    """

    models_dir = os.path.join(PROJECT_DIR, 'models')

    files = os.listdir(models_dir)
    filename = sorted(files)[-1]

    model = tf.keras.models.load_model(os.path.join(models_dir, filename))

    return model


def prediction(model, X_test, y_test):
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


def plot_conf_matrix(y_pred, y_test):
    """
    Generates a heatmap of the confusion matrix based on the predicted labels and the actual labels.

    Parameters:
        y_pred (array-like): An array or a list-like object containing the predicted labels.
        y_test (array-like): An array or a list-like object containing the actual labels.

    Returns:
        None
    """

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, alpha=0.7, linewidths=2, xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Heatmap of the Confusion Matrix', fontsize=18, fontweight='bold')

    output_dir = os.path.join(PROJECT_DIR, 'plots', 'confusion')
    filename = f"Confusion_Matrix.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

    plt.show()


def plot_metrics(y_pred, y_test):
    """
    Generates a lollipop plot of precision, F1 score, and recall based on the predicted labels and the actual labels.

    Parameters:
        y_pred (array-like): An array or a list-like object containing the predicted labels.
        y_test (array-like): An array or a list-like object containing the actual labels.
    Returns:
        None
    """

    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    metrics = {'Precision': precision, 'Recall': recall, 'F1 Score': f1}

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.vlines(x=list(metrics.keys()), ymin=0, ymax=list(metrics.values()), alpha=0.7)
    ax.plot(list(metrics.keys()), list(metrics.values()), "o")

    for i, metric in enumerate(metrics.keys()):
        ax.text(i, metrics[metric], f'{metrics[metric]:.2f}', ha='center', va='bottom')

    ax.set_title('Metrics', size=18, fontweight='bold')
    ax.grid(axis='y')

    output_dir = os.path.join(PROJECT_DIR, 'plots', 'metrics')
    filename = f"Metrics.png"
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

    model = create_model()
    model, history = train_model(model, X_train, y_train)
    plot_history(history)

    y_pred, y_test = prediction(model, X_test, y_test)
    plot_conf_matrix(y_pred, y_test)
    plot_metrics(y_pred, y_test)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocessing.main()

    main(X_train, X_test, y_train, y_test)
