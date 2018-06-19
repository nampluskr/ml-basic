import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 100, 0.01

# Data:
data = np.array([[1], [2], [3]], dtype=float)
target = np.array([[3], [5], [7]], dtype=float)

# Model:
weight = np.array([[0.1]])
bias = np.array([0.0])

# Training:
loss_train = []
for n in range(n_epoch):
    output = np.dot(data, weight) + bias
    loss = np.mean((output - target)**2)
    loss_train.append(loss)

    grad_output = 2*(output - target)/data.shape[0]
    grad_weight = np.dot(data.T, grad_output)
    grad_bias = np.sum(grad_output, axis=0)

    weight -= lr*grad_weight
    bias -= lr *grad_bias

# Evaluation:
print(weight, bias)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.plot(data, target, 'o')
ax1.plot(data, output, '--')
ax2.plot(loss_train)
plt.show()
