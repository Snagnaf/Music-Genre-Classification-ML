import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Normalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from variables import print_step, print_success

# CNN MODEL ARCHITECTURE ==================================================
def build_cnn(input_shape, num_classes, norm_layer=None):
    """
    Builds the CNN model architecture.

    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of genres to classify.
        norm_layer (keras.layers.Layer, optional): Normalization layer to include.

    Returns:
        keras.models.Sequential: The constructed CNN model.
    """
    model = Sequential()
    
    # Input Layer
    model.add(Input(shape=input_shape))
    
    # Normalization Layer (internal to the model)
    if norm_layer:
        model.add(norm_layer)
    
    # Conv Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Conv Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Conv Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Flatten & Dense
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# CNN MODEL TRAINING ==================================================
def train_cnn(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains a CNN (Convolutional Neural Network) model.

    Args:
        X_train: Training features (not standardized, dimensions will be added for CNN)
        y_train: Training labels (encoded)
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels

    Returns:
        tuple: (y_pred_cnn, best_cnn, history_cnn)
            - y_pred_cnn: Predictions on the test set
            - best_cnn: Best trained CNN model
            - history_cnn: Training history object
    """
    # Add Channel Dimension
    X_train_cnn = X_train[..., np.newaxis]
    X_val_cnn = X_val[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    # Prepare categorical labels for CNN
    # We fit on 'y_train' to learn the classes (Blues, Rock, etc.)
    y_train_cnn = to_categorical(y_train)
    y_val_cnn = to_categorical(y_val)
    y_test_cnn = to_categorical(y_test)

    # Create and adapt the normalization layer
    norm_layer = Normalization(axis=-1)
    norm_layer.adapt(X_train_cnn)

    # Build CNN model (with internal normalization)
    num_classes = y_train_cnn.shape[1]
    cnn = build_cnn(input_shape=(128, 130, 1), num_classes=num_classes, norm_layer=norm_layer)
    #cnn.summary()

    # Compile the model
    cnn.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Smart Callbacks
    callbacks = [
        # Stop if validation loss doesn't improve for 10 epochs
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),

        # Reduce Learning Rate if we get stuck (helps fine-tuning)
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001),

        # Save the BEST model (not the last one)
        ModelCheckpoint('src/cache/cnn_model.keras', monitor='val_accuracy', save_best_only=True)
    ]    

    # Train the model
    print_step("[MODULO 3] Training CNN...")
    history_cnn = cnn.fit(
        X_train_cnn, y_train_cnn,
        validation_data=(X_val_cnn, y_val_cnn),
        batch_size=32,
        epochs=50,  # -> EarlyStopping will handle the actual duration
        callbacks=callbacks,
        verbose=1
    )
    # Load Best Model
    best_cnn = tf.keras.models.load_model('src/cache/cnn_model.keras')
    print_success("✓ CNN model saved in cache.")

    # Evaluate
    loss_cnn, acc_cnn = best_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
    print_success(f"✓ Accuracy: {acc_cnn*100:.4f}%")

    # Confusion Matrix
    y_pred_cnn = np.argmax(best_cnn.predict(X_test_cnn), axis=1)
    y_proba_cnn = best_cnn.predict(X_test_cnn)
    y_true_cnn = np.argmax(y_test_cnn, axis=1)

    return y_pred_cnn, best_cnn, history_cnn, y_proba_cnn
