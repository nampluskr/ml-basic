import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


def cross_entropy(y, t):
    return -np.sum(t*np.log(y + 1.0E-8))/y.shape[0]


def accuracy(y, t):
    return ((y>0.5) == t).mean()


n_epoch, lr = 1000, 0.01

raw_data = np.genfromtxt("diabetes.csv", delimiter=',')
data = raw_data[:,:-1]
target = raw_data[:,-1].reshape(-1,1)

np.random.seed(0)
w1 = np.random.normal(0, 1, (8, 100))
b1 = np.zeros(100)
w2 = np.random.normal(0, 1, (100, 100))
b2 = np.zeros(100)
w3 = np.random.normal(0, 1, (100, 1))
b3 = np.zeros(1)

loss_train = []
score_train = []
for ep in range(n_epoch):
    lin1 = np.dot(data, w1) + b1
    act1 = sigmoid(lin1)
    lin2 = np.dot(act1, w2) + b2
    act2 = sigmoid(lin2)
    lin3 = np.dot(act2, w3) + b3
    output = sigmoid(lin3)
    loss = cross_entropy(output, target)
    loss_train.append(loss)
    score_train.append(accuracy(output, target))

    grad_lin3 = (output - target)/data.shape[0]
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

    w3 -= lr*grad_w3
    b3 -= lr*grad_b3
    w2 -= lr*grad_w2
    b2 -= lr*grad_b2
    w1 -= lr*grad_w1
    b1 -= lr*grad_b1

print("Epoch[%3d] >> Loss = %f, Acc. = %f" % (n_epoch,
      loss_train[-1], score_train[-1]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(loss_train)
ax2.plot(score_train)