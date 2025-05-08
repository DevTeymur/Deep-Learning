# Autoregressive Transformer: Predict 1 step at a time

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout,
    MultiHeadAttention, Add
)


mat = scipy.io.loadmat(r'Xtrain.mat')
data = mat['Xtrain'].flatten().reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

train_data = scaled_data[:800]
test_data = scaled_data[800:]

window_size = 50
predict_length = 200

X, y = [], []
for i in range(len(train_data) - window_size):
    X.append(train_data[i:i + window_size])
    y.append(train_data[i + window_size])

X = np.array(X)
y = np.array(y)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len, d_model):
        super().__init__()
        pos = np.arange(sequence_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :inputs.shape[1], :]


def transformer_block(inputs, num_heads=4, ff_dim=128):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
    x = transformer_block(x, num_heads=4, ff_dim=128)
    x = transformer_block(x, num_heads=4, ff_dim=128)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

model = build_transformer_model((window_size, 1))
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])


current_seq = train_data[-window_size:].reshape(1, window_size, 1)
predictions = []

for _ in range(predict_length):
    next_val = model.predict(current_seq, verbose=0)[0][0]
    predictions.append(next_val)
    next_step = np.array([[next_val]])
    current_seq = np.concatenate([current_seq[:, 1:, :], next_step.reshape(1, 1, 1)], axis=1)


true_test = test_data[:predict_length]
true_unscaled = scaler.inverse_transform(true_test)
pred_unscaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

mse = mean_squared_error(true_unscaled, pred_unscaled)
print("Mean Squared Error:", mse)


plt.figure(figsize=(12, 5))
plt.plot(true_unscaled, label='True')
plt.plot(pred_unscaled, label='Predicted')
plt.legend()
plt.title("Autoregressive Transformer 200-Step Forecast")
plt.grid(True)
plt.show()