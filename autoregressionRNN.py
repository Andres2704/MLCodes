import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

# Creating the time series for autoregression linear model problem
series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1
print(series.shape)
# Plotting time series
plt.plot(series)
plt.show()

# Building the dataset
T = 10 # windows length
X = []
Y = []

for t in range(len(series)-T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

X = np.array(X)
Y = np.array(Y)
N = len(X)
print('X.shape ', X.shape)
print('Y.shape ', Y.shape)



# Building the autoregressive model with RNN
i = Input(shape=(T,))
x = Dense(1)(i)

model = Model(i, x)
model.compile(optimizer=Adam(lr=0.1), loss='mse')

# Training the model using 50% of the data for training
r = model.fit(X[:-N//2], Y[:-N//2], validation_data=(X[-N//2:], Y[-N//2:]), epochs = 80)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='valloss')
plt.legend()

# Wrong forecast using true targets
validation_targe = Y[-N//2:]
val_predictions = []

# Index of first validation input
i = -N//2

while len(val_predictions) < len(validation_targe):
  p = model.predict(X[i].reshape(1,-1))[0,0]
  i += 1

  # Updating prediction list
  val_predictions.append(p)

plt.plot(validation_targe, label='forecast target')
plt.plot(val_predictions, label='forecast predictions')
plt.legend()

# Using the correct approach for forecast
validation_target = Y[-N//2:]
val_predictions = []

# Last value of train set
last = X[-N//2] 
while len(val_predictions) < len(validation_target):
  x_next = model.predict(last.reshape(1,-1))[0,0] # 1x1 array, an scalar

  # updating forecast list
  val_predictions.append(x_next)

  # Make the new input
  last = np.roll(last, -1) # Shift the values in a list to -1 to the left eg: a=[1,2,3] -> np.roll(a,-1)=[2,3,1]
  last[-1] = x_next # Att the last value in this array

plt.plot(validation_target[0:200], label='forecast target')
plt.plot(val_predictions[0:200], label='forecast predictions')
plt.legend()

