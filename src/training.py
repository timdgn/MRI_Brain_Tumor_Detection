from warnings import filterwarnings
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from constants import *
import preprocessing
import datetime
import os


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


def train_model(model, X, y):
    """
    Train a machine learning model using the specified data and training parameters.

    Parameters:
        model (object): The machine learning model to be trained.
        X (array-like): The input data for training the model.
        y (array-like): The target data for training the model.

    Returns:
        model (object): The trained machine learning model.
        history (object): The history object containing information about the training process.
    """

    print('Starting training...')

    # Compile the model with the specified loss, optimizer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Set up callbacks for TensorBoard, model checkpointing, and learning rate reduction
    tensorboard = TensorBoard(log_dir='logs')
    checkpoint = ModelCheckpoint("effnet.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto',
                                  verbose=1)

    # Train the model with the specified data and training parameters
    history = model.fit(X, y, validation_split=0.1, epochs=EPOCHS, verbose=1, batch_size=32,
                        callbacks=[tensorboard, checkpoint, reduce_lr])

    print('Training done...', end='\n\n')

    return model, history


def save_model(model):
    """
    Saves the given model to the specified directory.

    Args:
        model: The model to be saved.

    Returns:
        filename (str): The name of the saved file.
    """

    now = datetime.datetime.now()
    output_dir = os.path.join(os.getcwd(), '..', 'models')
    filename = f"effnet_{now.strftime('%Y-%m-%d_%H-%M-%S')}.h5"
    model.save(os.path.join(output_dir, filename))

    return filename


def plot_history(history):
    """
    Generate a plot of training and validation accuracy and loss over epochs.

    Parameters:
    - history: The training history object containing accuracy and loss values.

    Returns:
    None
    """

    filterwarnings('ignore')

    epochs = [i for i in range(EPOCHS)]
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    # fig.text(s='Epochs vs. Training and Validation Accuracy/Loss', size=18, fontweight='bold',
    #          fontname='monospace', color=COLORS_DARK[1], y=1, x=0.28, alpha=0.8)
    fig.suptitle('Epochs vs. Training and Validation Accuracy/Loss', fontsize=18, fontweight='bold')

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

    now = datetime.datetime.now()
    output_dir = os.path.join(os.getcwd(), '..', 'plots')
    filename = f"Accuracy_Loss_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)

    fig.show()


def main(X_train, y_train):
    """
    A function that takes in the training data and trains a model using it.

    Parameters:
    - X_train: numpy array, the training data.
    - y_train: numpy array, the target labels for the training data.

    Returns:
    - model_path: str, the path where the trained model is saved.
    """

    model = create_model()
    model, history = train_model(model, X_train, y_train)
    model_path = save_model(model)
    plot_history(history)

    return model_path


if __name__ == '__main__':

    X_train, _, y_train, _ = preprocessing.main()

    model_path = main(X_train, y_train)
