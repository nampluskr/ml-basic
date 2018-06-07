import numpy as np
import matplotlib.pyplot as plt

# Set hyper-parameters:
n_epoch, lr = 100, 0.01

# Load data: y = 2 x + 1
data = np.array([[1], [2], [3]], dtype=float)
target = np.array([[3], [5], [7]], dtype=float)

# Setup a model:
weight = np.array([[0.1]])
bias = 0.0

# Train the model:
loss_train = []
for epoch in range(n_epoch):
    # Forward propagation:
    output = np.dot(data, weight) + bias
    loss = np.mean((output - target)**2)
    loss_train.append(loss)

    # Backward propagation:
    grad_output = 2*(output - target)/data.shape[0]
    grad_weight = np.dot(data.T, grad_output)
    grad_bias = np.sum(grad_output, axis=0)

    # Update model parameters:
    weight -= lr*grad_weight
    bias -= lr*grad_bias

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data, target, 'o')
ax1.plot(data, output)
ax2.plot(loss_train)
