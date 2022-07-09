import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('C:\Code\Machine_Learning\Data_Science_projects\MSFT.csv',index_col='Date')
#print(data.head(3))

#-----Forecasting Project-----#
data = data[["Close"]]
print(data.head(2))
print(f"Shape: {data.shape}")

x = data.index.to_numpy()
y = data.Close.to_numpy()

# plt.figure()
# plt.plot(x,y)
# plt.show()
def get_labelled_window(y,horizon):
  return y[:,:-horizon],y[:,-horizon:]

def make_windows(y, window_size,horizon):
  window_step = np.expand_dims(np.arange(window_size+horizon),axis=0)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)),axis=0).T
  # print(f"Window indexes:\n {window_indexes, window_indexes.shape}")
  windowed_array = y[window_indexes]
  # print(f"windowed array {windowed_array,windowed_array.shape }")
  windows, labels = get_labelled_window(windowed_array, horizon)
  return windows,labels
full_windows, full_labels = make_windows(y,7,1)
print("Make Windows",len(full_windows), len(full_labels))

def make_train_test_splits(windows,labels, test_split=0.2):
  split_size = int(len(windows)*(1-test_split)) #this will default to 80%train /20% test
  train_windows = windows[:split_size]
  test_windows = windows[split_size:]
  train_labels = labels[:split_size]
  test_labels = labels[split_size:]
  return train_windows,test_windows,train_labels,test_labels

train_windows,test_windows,train_labels,test_labels=make_train_test_splits(full_windows,full_labels)
print("Train Windows:",train_windows.shape,train_labels.shape)
print(train_windows[0],train_labels[0])

import tensorflow as tf

# Set random seed for as reproducible results as possible
tf.random.set_seed(42)

#1. Construct model
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(1,activation='linear')
])
#2.Compile the model
model_1.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.mae,
                metrics=["mae"])
#3.Fit the model
model_1.fit(train_windows,train_labels,epochs=100,
            batch_size=128,
            validation_data=(test_windows,test_labels),
            callbacks=[tf.keras.callbacks.ModelCheckpoint('checkpoints/model_1'),
                        tf.keras.callbacks.EarlyStopping(patience=10)])