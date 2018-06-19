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


class Classifier:
    def __init__(self):
        np.random.seed(0)
        self.w1 = np.random.normal(0, 1, (30, 200))
        self.b1 = np.zeros(200)
        self.w2 = np.random.normal(0, 1, (200, 100))
        self.b2 = np.zeros(100)
        self.w3 = np.random.normal(0, 1, (100, 1))
        self.b3 = np.zeros(1)

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        lin1 = np.dot(data, self.w1) + self.b1
        self.act1 = sigmoid(lin1)
        lin2 = np.dot(self.act1, self.w2) + self.b2
        self.act2 = sigmoid(lin2)
        lin3 = np.dot(self.act2, self.w3) + self.b3
        self.output = sigmoid(lin3)
        return self.output

    def backward(self, dout=1):
        grad_lin3 = dout*(self.output - target)/data.shape[0]
        self.grad_w3 = np.dot(self.act2.T, grad_lin3)
        self.grad_b3 = np.sum(grad_lin3, axis=0)

        grad_act2 = np.dot(grad_lin3, self.w3.T)
        grad_lin2 = self.act2*(1-self.act2)*grad_act2
        self.grad_w2 = np.dot(self.act1.T, grad_lin2)
        self.grad_b2 = np.sum(grad_lin2, axis=0)

        grad_act1 = np.dot(grad_act2, self.w2.T)
        grad_lin1 = self.act1*(1-self.act1)*grad_act1
        self.grad_w1 = np.dot(data.T, grad_lin1)
        self.grad_b1 = np.sum(grad_lin1, axis=0)

    def update(self):
        self.w3 -= lr*self.grad_w3
        self.b3 -= lr*self.grad_b3
        self.w2 -= lr*self.grad_w2
        self.b2 -= lr*self.grad_b2
        self.w1 -= lr*self.grad_w1
        self.b1 -= lr*self.grad_b1


# Set hyper-parameters:
n_epoch, lr = 1000, 0.001

# Load data:
cancer = load_breast_cancer()
data, x_test, target, y_test = train_test_split(cancer.data,
                                    cancer.target, random_state=0)
target = target.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# Setup a model:
model = Classifier()
error = cross_entropy

# Train the model:
loss_train, loss_test = [], []
acc_train, acc_test = [], []
for ep in range(n_epoch):
    # Forward propagation:
    output = model(data)
    loss_train.append(error(output, target))
    acc_train.append(score(output, target))

    # Backward propagation:
    model.backward()

    # Update model parameters:
    model.update()

    # Evaluate the trained model:
    predict = model(x_test)
    loss_test.append(error(predict, y_test))
    acc_test.append(score(predict, y_test))


print("Epoch[%3d] >> Loss = %f, Acc. = %f" % (n_epoch,
      loss_train[-1], acc_train[-1]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(loss_train, label="Training")
ax1.plot(loss_test, label="Test")
ax1.legend()
ax1.set_title("Loss")
ax2.plot(acc_train, label="Trainig")
ax2.plot(acc_test, label="Test")
ax2.legend()
ax2.set_title("Accuracy")