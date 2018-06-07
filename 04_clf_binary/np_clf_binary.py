import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def cross_entropy(y, t):
    batch_size = y.shape[0] if y.ndim == 2 else 1
    return -np.sum(t*np.log(y + 1.0E-8))/batch_size


def score(y, t):
    return ((y>0.5) == t).mean()


# Set hyper-parameters:
n_epoch, lr = 100, 0.001

# Load data:
cancer = load_breast_cancer()
data, x_test, target, y_test = train_test_split(cancer.data,
                                    cancer.target, random_state=0)
target = target.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Setup a model:
np.random.seed(0)
w1 = np.random.normal(0, 1, (30, 200))
b1 = np.zeros(200)
w2 = np.random.normal(0, 1, (200, 100))
b2 = np.zeros(100)
w3 = np.random.normal(0, 1, (100, 1))
b3 = np.zeros(1)

# Train the model:
loss_train = []
acc_train = []
for ep in range(n_epoch):
    # Forward propagation:
    lin1 = np.dot(data, w1) + b1
    act1 = sigmoid(lin1)
    lin2 = np.dot(act1, w2) + b2
    act2 = sigmoid(lin2)
    lin3 = np.dot(act2, w3) + b3
    output = sigmoid(lin3)
    loss = cross_entropy(output, target)
    loss_train.append(loss)
    acc_train.append(score(output, target))

    # Backward propagation:
    grad_lin3 = (output - target)/data.shape[0]
    grad_w3 = np.dot(act2.T, grad_lin3)
    grad_b3 = np.sum(grad_lin3, axis=0)

    grad_act2 = np.dot(grad_lin3, w3.T)
    grad_lin2 = act2*(1-act2)*grad_act2
    grad_w2 = np.dot(act1.T, grad_lin2)
    grad_b2 = np.sum(grad_lin2, axis=0)

    grad_act1 = np.dot(grad_act2, w2.T)
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
print("Epoch[%3d] >> Loss = %f, Acc. = %f" % (n_epoch,
      loss_train[-1], acc_train[-1]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(loss_train)
ax2.plot(acc_train)