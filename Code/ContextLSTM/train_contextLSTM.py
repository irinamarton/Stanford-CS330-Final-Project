import numpy as np
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import classification_report

import contextLSTM_util as util

#print("GPUs Available", len(tf.config.list_physical_devices('GPU')))

start = "2011-01-01"
end = "2023-06-30"
history = 10
test_values = 251


ticker_dict, tickerTest = util.get_categorical_tickers()
tickerList = ticker_dict['all']
tickerList = [item for sublist in tickerList for item in sublist]

prices_raw = util.get_ticker_values(tickerList, start, end)
prices_test = prices_raw[0].copy()
prices_raw = np.delete(prices_raw, 0)  #remove test ticker from training


y_train = []
x_train = []
for ticker in prices_raw:
    if ticker.size < test_values:
        continue
    price, label = util.data_preprocess_HL_labels(ticker[:-test_values], 0, None, history)  #w context
    #context_price, context_label = util.data_w_context(price, label)  # w context
    #y_train = np.concatenate((y_train, context_label))  # w context
    price, label = util.data_preprocess_HL_labels(ticker[:-test_values], 0, None, history)  #w/o context
    y_train = np.concatenate((y_train, label))  #w/o context
    if len(x_train) == 0:
        #x_train = context_price  # w context
        x_train = price  # w/o context
    else:
        #x_train = np.concatenate((x_train, context_price), axis=0)  # w context
        x_train = np.concatenate((x_train, price), axis=0)  # w/o context

for_test = prices_test[-test_values-history:]  #w context
for_test = prices_test[-(test_values-1)-history:]  #w/o context
#x_t, y_t = util.data_preprocess_HL_labels(for_test, 0, None, history)  #w context
#x_test, y_test = util.data_w_context(x_t, y_t)  #w context
x_test, y_test = util.data_preprocess_HL_labels(for_test, 0, None, history)  #w/o context

model = Sequential()
#with tf.device('/GPU:0'):
model.add(LSTM(units=20, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=20, return_sequences=True))
model.add(LSTM(units=20, return_sequences=True))
model.add(LSTM(units=20))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, steps_per_epoch=200,
          validation_steps=50, verbose=1)

predictions = model.predict(x_test)
predicted_classed = np.argmax(predictions, axis=1)
_, ideal = util.data_preprocess_HL_labels(prices_test[-(test_values-1)-history:], 0, None, history)
#print("IDEAL", len(ideal), len(predicted_classed), ideal, predicted_classed)
print(classification_report(ideal, predicted_classed))
print("FIRST 50", classification_report(ideal[:50], predicted_classed[:50]))

util.plot_bot_decision(prices_test[-(test_values-1):][:50], ideal[:50], predicted_classed[:50])
