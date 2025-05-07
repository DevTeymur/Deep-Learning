# LSTM Sequence-to-Sequence + Ensemble Forecasting

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, RepeatVector


mat = scipy.io.loadmat('Xtrain.mat')
data = mat['Xtrain'].flatten().reshape(-1, 1)

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
y = np.array(y)
y = y.reshape((y.shape[0], y.shape[1], 1))


def build_seq2seq_model(input_shape, output_length):
    inputs = Input(shape=input_shape)
    x = LSTM(64)(inputs)
    x = RepeatVector(output_length)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = TimeDistributed(Dense(1))(x)
    return Model(inputs, x)


n_models = 5
models = []
predictions = []

for i in range(n_models):
    model = build_seq2seq_model((window_size, 1), predict_length)
    model.compile(optimizer='adam', loss='mse')
    print(f"Training model {i+1}/{n_models}...")
    model.fit(X, y, epochs=50, batch_size=32, verbose=0,
              validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    models.append(model)


    input_seq = train_data[-window_size:].reshape(1, window_size, 1)
    pred = model.predict(input_seq, verbose=0).reshape(-1)
    predictions.append(pred)


predictions = np.array(predictions)
avg_prediction = np.mean(predictions, axis=0)


true_test = test_data[:predict_length]
true_unscaled = scaler.inverse_transform(true_test)
pred_unscaled = scaler.inverse_transform(avg_prediction.reshape(-1, 1))

mse = mean_squared_error(true_unscaled, pred_unscaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_unscaled, pred_unscaled)

print("Ensemble Seq2Seq Results:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)


plt.figure(figsize=(12, 5))
plt.plot(true_unscaled, label='True')
plt.plot(pred_unscaled, label='Predicted')
plt.legend()
plt.title("Seq2Seq LSTM Ensemble 200-Step Forecast")
plt.grid(True)
plt.show()

