import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, UpSampling1D, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import joblib


def unet_1d(input_size):
    num_classes = 2
    inputs = Input(input_size)

    # Encoder (Downsampling)
    conv1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    # Bottleneck
    conv3 = Conv1D(256, 3, activation='relu', padding='same')(pool2)

    # Decoder (Upsampling)
    up4 = UpSampling1D(size=2)(conv3)
    conv4 = Conv1D(128, 3, activation='relu', padding='same')(up4)
    merge4 = Concatenate()([conv2, conv4])

    up5 = UpSampling1D(size=2)(merge4)
    conv5 = Conv1D(64, 3, activation='relu', padding='same')(up5)
    merge5 = Concatenate()([conv1, conv5])

    # Add GlobalAveragePooling1D to reduce spatial dimensions
    gap = GlobalAveragePooling1D()(merge5)

    # Output layer for multi-class classification (28 classes)
    outputs = Dense(num_classes, activation='softmax')(gap)

    model = Model(inputs, outputs)
    return model

def load_and_preprocess_data(data_file, target_column, num_classes=28):
    # Load the dataset
    data = pd.read_csv(data_file)

    print(data.head())

    data['Win %'] = data['Win %'].str.replace('%', '').astype(float) / 100
    data['Pick %'] = data['Pick %'].str.replace('%', '').astype(float) / 100
    data['Dmg/Round'] = data['Dmg/Round'].astype(float)
    data['KDA'] = data['KDA'].astype(float)

    # Define features and target
    features = data[['Score', 'Trend', 'Pick %', 'Dmg/Round', 'KDA']]
    features = pd.get_dummies(features, columns=['Trend'], drop_first=True)
    target = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Convert to NumPy arrays
    X_train = X_train.to_numpy().astype('float32')
    X_test = X_test.to_numpy().astype('float32')
    y_train = to_categorical(y_train, num_classes=num_classes).astype('float32')
    y_test = to_categorical(y_test, num_classes=num_classes).astype('float32')

    return X_train, X_test, y_train, y_test

def train_unet(data_file, target_column, batch_size=32, epochs=50, learning_rate=0.001):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_file, target_column)

    # Reshape the input for Conv1D
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Define the model
    input_shape = (X_train.shape[1], 1)
    model = unet_1d(input_shape)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

    # Train the model
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), 
              epochs=epochs, batch_size=batch_size, 
              callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
    print(f'Test accuracy: {test_acc}')
    return model


if __name__ == "__main__":
    data_file = 'data_clean.csv' 
    target_column = 'Win %'        
    models = train_unet(data_file, target_column)
    joblib.dump(models, 'model.pkl')
