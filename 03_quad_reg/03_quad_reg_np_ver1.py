import numpy as np
import keras
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 1000, 0.001

# Data: y = x^2 + 3
np.random.seed(10)
noise = np.random.normal(0, 1, (1000, 1))
data = np.random.uniform(-10, 10, (1000,1))
target = data**2 + 3 + noise

# Model:
model = keras.models.Sequential()
model.add(keras.layers.Dense(200, activation='sigmoid', input_shape=(1,)))
model.add(keras.layers.Dense(100, activation='sigmoid'))
model.add(keras.layers.Dense(1))
model.summary()

model.compile(loss='mse', optimizer=keras.optimizers.sgd(lr=lr))

# Training:
history = model.fit(data, target, epochs=n_epoch, verbose=0)

# Evaluation:
plt.plot(history.history['loss'])