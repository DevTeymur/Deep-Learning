import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Add

# --- Load and scale data ---
mat = scipy.io.loadmat('Xtrain.mat')
data = mat['Xtrain']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

train_data = scaled_data[:800]
test_data = scaled_data[800:]

window_size = 50
predict_length = 200

X, y = [], []
for i in range(len(train_data) - window_size - predict_length + 1):
    X.append(train_data[i:i + window_size])
    y.append(train_data[i + window_size:i + window_size + predict_length])
X = np.array(X)
y = np.array(y[:, 0])  # simplify: predict just the next value for now

X = X.reshape((X.shape[0], window_size, 1))

# --- Positional Encoding Layer ---
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

# --- Transformer block ---
def transformer_block(inputs, num_heads, ff_dim):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attn_output = Dropout(0.1)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([inputs, attn_output]))

    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(Add()([out1, ffn_output]))

# --- Build Transformer model ---
def build_transformer_model(input_shape, num_heads=2, ff_dim=64):
    inputs = Input(shape=input_shape)
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)
    x = transformer_block(x, num_heads, ff_dim)
    x = transformer_block(x, num_heads, ff_dim)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

model = build_transformer_model((window_size, 1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# --- Train ---
model.fit(X, y, epochs=20, batch_size=32)

# --- Predict ---
current_seq = train_data[-window_size:].flatten().tolist()
predictions = []

for _ in range(predict_length):
    input_seq = np.array(current_seq[-window_size:]).reshape(1, window_size, 1)
    next_pred = model.predict(input_seq, verbose=0)[0][0]
    predictions.append(next_pred)
    current_seq.append(next_pred)

# --- Evaluate ---
true_test = test_data[:predict_length].flatten()
predicted = np.array(predictions).flatten()

mse = mean_squared_error(true_test, predicted)
print("Predicted next 200 values:")
print(predicted)
print("Mean Squared Error:", mse)
