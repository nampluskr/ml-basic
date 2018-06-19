import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 100, 0.01

# Data:
data = np.array([1, 2, 3])
target = np.array([3, 5, 7])

# Model:
weight = 0.1
bias = 0.0

# Training:
loss_train = []
for n in range(n_epoch):
    output = weight*data + bias
    loss = np.mean((output - target)**2)
    loss_train.append(loss)

    grad_output = 2*(output - target)/data.shape[0]
    grad_weight = np.dot(grad_output, data)
    grad_bias = np.sum(grad_output)

    weight -= lr*grad_weight
    bias -= lr *grad_bias

# Evaluation:
print(weight, bias)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.plot(data, target, 'o')
ax1.plot(data, output, '--')
ax2.plot(loss_train)
plt.show()