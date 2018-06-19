import numpy as np
import keras
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 100, 0.01

# Data:
data = np.array([[1], [2], [3]])
target = np.array([[3], [5], [7]])

# Model:
model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.summary()
model.compile(loss='mse', optimizer=keras.optimizers.sgd(lr=lr))

# Training:
history = model.fit(data, target, epochs=n_epoch, verbose=0)

# Evaluation:
plt.plot(history.history['loss'])