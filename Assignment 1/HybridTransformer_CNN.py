import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Load and preprocess data ---
train_data = scipy.io.loadmat('Xtrain.mat')['Xtrain'].flatten().reshape(-1, 1)
test_data = scipy.io.loadmat('Xtest.mat')['Xtest'].flatten().reshape(-1, 1)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

window_size = 50

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

# --- Positional Encoding Layer ---
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        pos = np.arange(sequence_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# --- Build Hybrid Model ---
def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding='causal', activation='relu')(inputs)
    x = layers.LayerNormalization()(x)
    x = PositionalEncoding(input_shape[0], 64)(x)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Bound output between 0 and 1
    return models.Model(inputs, outputs)

model = build_model((window_size, 1))
model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mae'])

# --- Train Model ---
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1)

# --- Predict and Evaluate ---
predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, 1)  # Ensure values are within [0,1]
predictions_inv = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"✅ MSE: {mse:.4f} | MAE: {mae:.4f}")

# --- Plot Results ---
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual', linewidth=2)
plt.plot(predictions_inv, label='Predicted', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
