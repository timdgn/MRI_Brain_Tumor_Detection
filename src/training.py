import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import specificity_score

import preprocessing as pre
from constants import *


def get_available_devices():
    """
    Get information about available devices.

    Parameters:
        None

    Returns:
        None
    """

    print("Number of CPU/GPU Available: ", len(tf.config.list_physical_devices()))
    print("CPU: ", tf.config.list_physical_devices('CPU'))
    print("GPU: ", tf.config.list_physical_devices('GPU'), end='\n\n')


def create_model():
    """
    Create EfficientNet model with specified input shape and output size.

    Returns:
        model (object): The trained machine learning model.
    """

    with tf.device('/GPU:0'):  # Replace with '/CPU:0' if you want to use CPU

        # Create EfficientNetB0 model with pre-trained weights
        effnet = tf.keras.applications.EfficientNetV2B0(weights='imagenet', include_top=False,
                                                        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        model = tf.keras.layers.GlobalAveragePooling2D()(effnet.output)
        model = tf.keras.layers.Dropout(rate=0.5)(model)
        model = tf.keras.layers.Dense(4, activation='softmax')(model)
        model = tf.keras.models.Model(inputs=effnet.input, outputs=model)

        return model


def train_model(model, X_train, X_val, y_train, y_val):
    """
    Train a machine learning model using the specified data and training parameters.

    Parameters:
        model (object): The machine learning model to be trained.
        X_train (array-like): The input data for training the model.
        y_train (array-like): The target data for training the model.
        X_val (array-like): The validation input data.
        y_val (array-like): The validation target data.

    Returns:
        history (object): The history object containing information about the training process.
    """

    print('\nStarting training...')

    # Compile the model with the specified loss, optimizer, and metrics
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy',
                           tf.keras.metrics.F1Score(average='macro', threshold=None, name='f1', dtype=None)])

    model.summary()

    now = datetime.datetime.now()
    output_dir = os.path.join(PROJECT_DIR, 'models')
    filename = f"effnet_{now.strftime('%Y-%m-%d_%H-%M-%S')}"

    # Save model architecture
    with open(os.path.join(output_dir, f"{filename}.json"), "w") as json_file:
        json_file.write(model.to_json())

    # Setting up callbacks model checkpointing, learning rate reduction, and F1 score
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', factor=0.3, patience=2, min_delta=0.001,
                                                     mode='max', verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(output_dir, f"{filename}.h5"), monitor="val_f1",
                                                    save_best_only=True, save_weights_only=True, mode='max', verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_f1', min_delta=0.001, patience=5, verbose=1, mode='max',
                                                  start_from_epoch=3)

    # Train the model with the specified data and training parameters
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE,
                        verbose=1, callbacks=[reduce_lr, checkpoint, early_stop])

    return model, history


def plot_history(history):
    """
    Generate a plot to visualize the training and validation F1 Score/loss over epochs and save the plots

    Parameters:
        history: A dictionary containing the training and validation F1 Score/loss/accuracy history.

    Returns:
        None
    """

    epochs = [ep + 1 for ep in history.epoch]
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    train_color = '#440154'
    val_color = '#5ec962'
    arrow_color = 'black'

    # Plot training & validation accuracy values
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history.history['accuracy'], color=train_color)
    plt.plot(epochs, history.history['val_accuracy'], color=val_color)
    plt.title('Model Accuracy', size=18, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Annotate the highest point for accuracy
    max_acc = max(history.history['accuracy'])
    max_val_acc = max(history.history['val_accuracy'])
    plt.annotate(
        f'Max Train Accuracy: {format(max_acc, ".3f")}\n(Epoch {history.history["accuracy"].index(max_acc) + 1})',
        xy=(history.history['accuracy'].index(max_acc) + 1, max_acc), xytext=(10, -60),
        textcoords='offset points', arrowprops=dict(arrowstyle='->', color=arrow_color))
    plt.annotate(
        f'Max Validation Accuracy: {format(max_val_acc, ".3f")}\n(Epoch {history.history["val_accuracy"].index(max_val_acc) + 1})',
        xy=(history.history['val_accuracy'].index(max_val_acc) + 1, max_val_acc), xytext=(10, -80),
        textcoords='offset points', arrowprops=dict(arrowstyle='->', color=arrow_color))

    # Plot training & validation f1 values
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history.history['f1'], color=train_color)
    plt.plot(epochs, history.history['val_f1'], color=val_color)
    plt.title('Model F1 Score', size=18, fontweight='bold')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Annotate the highest point for F1 Score
    max_f1 = max(history.history['f1'])
    max_val_f1 = max(history.history['val_f1'])
    plt.annotate(f'Max Train F1 Score: {format(max_f1, ".3f")}\n(Epoch {history.history["f1"].index(max_f1) + 1})',
                 xy=(history.history['f1'].index(max_f1) + 1, max_f1), xytext=(10, -60),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color=arrow_color))
    plt.annotate(f'Max Validation F1 Score: {format(max_val_f1, ".3f")}\n(Epoch {history.history["val_f1"].index(max_val_f1) + 1})',
                 xy=(history.history['val_f1'].index(max_val_f1) + 1, max_val_f1), xytext=(10, -80),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color=arrow_color))

    # Plot training & validation loss values
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history.history['loss'], color=train_color)
    plt.plot(epochs, history.history['val_loss'], color=val_color)
    plt.title('Model Loss', size=18, fontweight='bold')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Annotate the lowest point for loss
    min_loss = min(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    plt.annotate(f'Min Train Loss: {format(min_loss, ".3f")}\n(Epoch {history.history["loss"].index(min_loss) + 1})',
                 xy=(history.history['loss'].index(min_loss) + 1, min_loss), xytext=(10, 40),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color=arrow_color))
    plt.annotate(f'Min Validation Loss: {format(min_val_loss, ".3f")}\n(Epoch {history.history["val_loss"].index(min_val_loss) + 1})',
                 xy=(history.history['val_loss'].index(min_val_loss) + 1, min_val_loss), xytext=(10, 60),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color=arrow_color))

    plt.tight_layout()

    # Saving the plot
    output_dir = os.path.join(PROJECT_DIR, 'plots', 'history')
    filename = f'History.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

    plt.show()


def load_last_model():
    """
    Loads the last model created from the specified folder.

    Returns:
        model: The loaded model.
    """

    models_dir = os.path.join(PROJECT_DIR, 'models')

    # Find the latest JSON file
    json_files = [f for f in os.listdir(models_dir) if f.endswith('.json')]
    latest_json_file = sorted(json_files)[-1]

    # Load the model architecture from JSON
    with open(os.path.join(models_dir, latest_json_file), "r") as json_file:
        loaded_model_json = json_file.read()

    model = tf.keras.models.model_from_json(loaded_model_json)

    # Load the weights from the corresponding H5 file
    weights_file = latest_json_file.replace('.json', '.h5')
    model.load_weights(os.path.join(models_dir, weights_file))

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
        y_pred (array-like): An array object containing the predicted labels.
        y_test (array-like): An array object containing the actual labels.

    Returns:
        None
    """

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', alpha=0.7, linewidths=2, xticklabels=LABELS, yticklabels=LABELS,
                cmap='viridis')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Heatmap of the Test dataset Confusion Matrix', fontsize=18, fontweight='bold')

    output_dir = os.path.join(PROJECT_DIR, 'plots', 'confusion')
    filename = f'Confusion_Matrix.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

    plt.show()


def plot_metrics(y_pred, y_test):
    """
    Generates a bar plot of accuracy, precision, recall (sensitivity), specificity, and F1 score based on the predicted labels and the actual labels.

    Parameters:
        y_pred (array-like): An array object containing the predicted labels.
        y_test (array-like): An array object containing the actual labels.

    Returns:
        None
    """

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')  # Sensitivity
    specificity = specificity_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1 Score': f1
    }

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), hue=list(metrics.keys()), palette="viridis",
                     legend=False)

    # Add the numbers above the bars
    for i, metric in enumerate(metrics.keys()):
        ax.text(i, metrics[metric] + 0.01, f'{metrics[metric]:.3f}', ha='center', va='bottom', fontsize=10)

    plt.title('Test dataset metrics', size=18, fontweight='bold')
    plt.ylabel('Value')
    plt.tight_layout()

    # Make the bars thinner
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - 0.5
        patch.set_width(0.5)
        patch.set_x(patch.get_x() + diff * .5)

    output_dir = os.path.join(PROJECT_DIR, 'plots', 'metrics')
    filename = 'Metrics.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

    plt.show()


def main(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Trains a model with the provided training data, saves the trained model,
    and generates predictions and a confusion matrix.

    Parameters:
        X_train (numpy.ndarray): Training data features.
        X_val (numpy.ndarray): Validation data features.
        X_test (numpy.ndarray): Test data features.
        y_train (numpy.ndarray): Training data labels.
        y_val (numpy.ndarray): Validation data labels.
        y_test (numpy.ndarray): Test data labels.

    Returns:
        None
    """

    get_available_devices()

    model = create_model()

    start_time = time.perf_counter()
    model, history = train_model(model, X_train, X_val, y_train, y_val)
    time_taken = time.perf_counter() - start_time
    print(f'--- Training time taken: {time_taken:.2f} seconds ({time_taken / 60:.2f} minutes) ---')

    plot_history(history)

    y_pred, y_test = prediction(model, X_test, y_test)
    plot_conf_matrix(y_pred, y_test)
    plot_metrics(y_pred, y_test)


if __name__ == '__main__':

    X_train, X_val, X_test, y_train, y_val, y_test = pre.preprocessing()

    main(X_train, X_val, X_test, y_train, y_val, y_test)
