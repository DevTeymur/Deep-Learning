import scipy.io
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

mat = scipy.io.loadmat('Xtrain.mat')
data = mat['Xtrain']

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 800 for train, 200 for test
train_data = scaled_data[:800]
test_data = scaled_data[800:]

window_size = 50  # number of past steps to use
predict_length = 200

X, y = [], []
for i in range(len(train_data) - window_size - predict_length + 1):
    X.append(train_data[i:i + window_size])
    y.append(train_data[i + window_size:i + window_size + predict_length])
X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=20, batch_size=32)

# Predicting next 200 values 1 value at a time
current_seq = train_data[-window_size:].flatten().tolist()
predictions = []

for _ in range(predict_length):
    input_seq = np.array(current_seq[-window_size:]).reshape(1, window_size, 1)
    next_pred = model.predict(input_seq)[0][0] 
    predictions.append(next_pred)
    current_seq.append(next_pred)

print("Predicted next 200 values:")
print(predictions)

from sklearn.metrics import mean_squared_error
true_test = test_data[:200].flatten()
predicted = np.array(predictions).flatten()

mse = mean_squared_error(true_test, predicted)
print("Mean Squared Error:", mse)