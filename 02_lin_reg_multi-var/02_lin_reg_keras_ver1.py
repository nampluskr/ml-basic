import numpy as np
import keras
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 100, 1.0E-6

# Data:
data = np.array([[73, 80, 75], [93, 88, 93], [89, 91, 90],
                 [96, 98, 100], [73, 66, 70]], dtype=float)
target = np.array([[152], [185], [180], [196], [142]], dtype=float)

# Model:
model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(3,)))
model.summary()
model.compile(loss='mse', optimizer=keras.optimizers.sgd(lr=lr))

# Training:
history = model.fit(data, target, epochs=n_epoch, verbose=0)

# Evaluation:
plt.plot(history.history['loss'])