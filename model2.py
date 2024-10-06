import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the dataset
data = pd.read_csv('data_clean.csv')  # Replace with your CSV file path

# Preprocess the data
# Convert percentage columns to numeric
data['Win %'] = data['Win %'].str.replace('%', '').astype(float) / 100
data['Pick %'] = data['Pick %'].str.replace('%', '').astype(float) / 100
data['Attacker Win %'] = data['Attacker Win %'].str.replace('%', '').astype(float) / 100
data['Defender Win %'] = data['Defender Win %'].str.replace('%', '').astype(float) / 100

# Convert other numeric columns
data['Dmg/Round'] = data['Dmg/Round'].astype(float)
data['KDA'] = data['KDA'].astype(float)

# Define features and target
features = data[['Score', 'Trend', 'Pick %', 'Dmg/Round', 'KDA', 'Role', 'Tier', 'Map', 'Attacker Win %', 'Defender Win %']]
features = pd.get_dummies(features, columns=['Trend', 'Role', 'Tier', 'Map'], drop_first=True)  # One-Hot Encoding

target = data['Win %']

# Normalize the data
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Reshape the data for CNN input (ต้องมีรูปแบบ [ตัวอย่าง, ความกว้าง, ช่อง])
X = np.expand_dims(features, axis=2)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=32, kernel_size=2, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer for regression (no activation function)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
ealy_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
epoch = 10
history = model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_data=(X_test, y_test), callbacks=ealy_stopping)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# Plot training & validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Actual vs Predicted Win %')
plt.xlabel('Actual Win %')
plt.ylabel('Predicted Win %')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.show()

# Save the model
model.save('cnn_model.h5')
