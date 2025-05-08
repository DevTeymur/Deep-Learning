import time
import scipy.io
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

from warnings import filterwarnings
filterwarnings('ignore')

mat = scipy.io.loadmat('Xtrain.mat')
data = mat['Xtrain']
print(len(data), data.shape)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 800 for train, 200 for test
train_data = scaled_data[:800]
test_data = scaled_data[800:]

predict_length = 200

def reshape_data(win_size, train_data=train_data, predict_length=predict_length):
    print(f"Reshaping data for window size {win_size}...", end=" ")
    X, y = [], []
    for i in range(len(train_data) - win_size - predict_length + 1):
        X.append(train_data[i:i + win_size])
        # y.append(train_data[i + win_size:i + win_size + predict_length])
        # y.append(train_data[i])
        y.append(train_data[i + win_size])  # Not a range

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print("done")
    return X, y


def compile_and_predict(win_size, logs=False):
    X, y = reshape_data(win_size)
    print(f"Compiling the model...", end=" ")
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(win_size, 1)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    print("done")

    print(f"Predicting next {predict_length} values...", end=" ")
    # Predicting next 200 values 1 value at a time
    current_seq = train_data[-win_size:].flatten().tolist()
    predictions = []

    for _ in range(predict_length):
        input_seq = np.array(current_seq[-win_size:]).reshape(1, win_size, 1)
        next_pred = model.predict(input_seq, verbose=0)[0][0] 
        predictions.append(next_pred)
        current_seq.append(next_pred)
    print("done")
    print("Predicted next 200 values:") if logs else None
    print(predictions) if logs else None
    return predictions

def compute_metrics(predictions, win_size, test_data=test_data):
    print("Computing metrics...", end=" ")
    true_test = test_data[:200].flatten()
    predicted = np.array(predictions).flatten()

    mse = mean_squared_error(true_test, predicted)
    mae = mean_absolute_error(true_test, predicted)
    print(f"For window size {win_size}:")
    print("Mean Squared Error:", round(mse, 4))
    print("Mean Absolute Error:", round(mae, 4))
    print("____"*15)
    return mse, mae


def get_time():
    """This function returns the current time in day-month hour:minute format for saving the results

    Returns:
        time: Current time in day-month hour:minute format
    """
    # Take time in day/monthh hour:minute format
    current_time = time.localtime()
    return time.strftime("%d-%m_%H:%M", current_time)


def plot_results(win_sizes, mses, maes):
    """
    Plots the MSE and MAE for different window sizes.
    """
    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel('Window Size')
    ax1.set_ylabel('MSE', color=color1)
    ax1.plot(win_sizes, mses, color=color1, label='MSE')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # Create a second y-axis
    color2 = 'tab:orange'
    ax2.set_ylabel('MAE', color=color2)
    ax2.plot(win_sizes, maes, color=color2, label='MAE')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('MSE and MAE vs Window Size')
    fig.tight_layout()
    plt.savefig(f'results/LSTM_result_{get_time()}.png')
    plt.close()

def plot_real_vs_pred(predictions, win_size):
    """
    Functio to plot the predicted values vs the true values for the first 200 values of the test set.
    """
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    true_test = scaler.inverse_transform(test_data[:200].reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(true_test, label='True Values')
    plt.plot(predictions, label='Predicted Values')
    plt.title(f'Predicted vs True Values for Window Size {win_size}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'results/LSTM_real_vs_pred_{win_size}_{get_time()}.png')
    plt.close()


window_sizes = np.arange(40, 71, 5)
print("Window sizes:", window_sizes)
win_sizes, mses, maes = [], [], []

for i in window_sizes:
    preds = compile_and_predict(i, logs=False)
    mse, mae = compute_metrics(preds, i)
    win_sizes.append(i)
    mses.append(mse)
    maes.append(mae)
    plot_real_vs_pred(preds, i)

# Plot the metrics
plot_results(win_sizes, mses, maes)

# Save the results as csv file
results = pd.DataFrame({'Window Size': win_sizes, 'MSE': mses, 'MAE': maes})
results.to_csv(f'results/LSTM_results_{get_time()}.csv', index=False)

