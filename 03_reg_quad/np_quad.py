import numpy as np
import matplotlib.pyplot as plt

# Set hyper-parameters:
n_epoch, lr = 1000, 0.001

# Load data: y = x^2 + 3
np.random.seed(10)
noise = np.random.normal(0, 1, (1000, 1))
data = np.random.uniform(-10, 10, (1000,1))
target = data**2 + 3 + noise

# Setup a model:
w1 = np.random.normal(0, 1, (1,200))
b1 = np.zeros(200)
w2 = np.random.normal(0, 1, (200,100))
b2 = np.zeros(100)
w3 = np.random.normal(0, 1, (100,1))
b3 = np.zeros(1)

# Train the model:
loss = np.zeros(n_epoch)
for epoch in range(n_epoch):
    # Forward propagation:
    lin1 = np.dot(data, w1) + b1
    act1 = 1/(1 + np.exp(-lin1))
    lin2 = np.dot(act1, w2) + b2
    act2 = 1/(1 + np.exp(-lin2))
    output = np.dot(act2, w3) + b3
    loss[epoch] = np.mean((output - target)**2)

    # Backward propagation:
    grad_lin3 = 2*(output - target)/data.shape[0]
    grad_w3 = np.dot(act2.T, grad_lin3)
    grad_b3 = np.sum(grad_lin3, axis=0)

    grad_act2 = np.dot(grad_lin3, w3.T)
    grad_lin2 = act2*(1-act2)*grad_act2
    grad_w2 = np.dot(act1.T, grad_lin2)
    grad_b2 = np.sum(grad_lin2, axis=0)

    grad_act1 = np.dot(grad_lin2, w2.T)
    grad_lin1 = act1*(1-act1)*grad_act1
    grad_w1 = np.dot(data.T, grad_lin1)
    grad_b1 = np.sum(grad_lin1, axis=0)

    # Update model parameters:
    w3 -= lr*grad_w3
    b3 -= lr*grad_b3
    w2 -= lr*grad_w2
    b2 -= lr*grad_b2
    w1 -= lr*grad_w1
    b1 -= lr*grad_b1

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss[epoch]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data, target, 'o')
ax1.plot(data, output, '.')
ax2.plot(loss)
