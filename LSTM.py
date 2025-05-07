import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# --- Load and preprocess data ---
mat = scipy.io.loadmat(r'C:\Users\guill\Downloads\Xtrain.mat')
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
y_1step = y[:, 0].reshape(-1, 1)


def build_simple_lstm(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64)(inputs)
    x = Dense(1)(x)
    return Model(inputs, x)

model = build_simple_lstm((window_size, 1))
model.compile(optimizer='adam', loss='mse')


model.fit(X, y_1step, epochs=50, batch_size=32, verbose=1,
          validation_split=0.1,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])


input_seq = train_data[-window_size:].reshape(1, window_size, 1)
preds = []

for _ in range(predict_length):
    next_val = model.predict(input_seq, verbose=0)
    preds.append(next_val[0, 0])
    input_seq = np.append(input_seq[:, 1:, :], [[[next_val[0, 0]]]], axis=1)

prediction = np.array(preds).reshape(-1, 1)


true_test = test_data[:predict_length]
true_unscaled = scaler.inverse_transform(true_test)
pred_unscaled = scaler.inverse_transform(prediction)

mse = mean_squared_error(true_unscaled, pred_unscaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_unscaled, pred_unscaled)

print("1-Step Iterative LSTM Results:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)


plt.figure(figsize=(12, 5))
plt.plot(true_unscaled, label='True')
plt.plot(pred_unscaled, label='Predicted')
plt.legend()
plt.title("1-Step Iterative LSTM: 200-Step Forecast")
plt.grid(True)
plt.show()


residuals = true_unscaled - pred_unscaled
plt.figure(figsize=(12, 3))
plt.plot(residuals, color='gray')
plt.title("Residuals (True - Predicted)")
plt.grid(True)
plt.show()
