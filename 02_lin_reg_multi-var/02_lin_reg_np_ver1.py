import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 100, 1.0E-6

# Data:
data = np.array([[73, 80, 75], [93, 88, 93], [89, 91, 90],
                 [96, 98, 100], [73, 66, 70]], dtype=float)
target = np.array([[152], [185], [180], [196], [142]], dtype=float)

# Model:
weight = np.random.rand(3,1)*0.1
bias = np.zeros(1)

# Training:
loss_train = []
for epoch in range(n_epoch):
    output = np.dot(data, weight) + bias
    loss = np.mean((output - target)**2)
    loss_train.append(loss)

    grad_output = 2*(output - target)/float(data.shape[0])
    grad_weight = np.dot(data.T, grad_output)
    grad_bias = np.sum(grad_output, axis=0)

    weight -= lr*grad_weight
    bias -= lr*grad_bias

# Evaluation:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data, target, 'o')
ax1.plot(data, output, 'x')
ax2.plot(loss_train)
