import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def score(y, t):
    return torch.eq(y>0.5, t>0.5).float().mean()

gpu = 1
device = torch.device("cuda") if gpu else torch.device("cpu")

# Set hyper-parameters:
n_epoch, lr = 10000, 1.0E-5

# Load data:
cancer = load_breast_cancer()
data, x_test, target, y_test = train_test_split(cancer.data,
                                    cancer.target, random_state=0)
data = torch.from_numpy(data).float().to(device)
target = torch.from_numpy(target).float().view(-1,1).to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_test = torch.from_numpy(y_test).float().view(-1,1).to(device)

# Setup a model:
torch.manual_seed(0)
model = torch.nn.Sequential(
        torch.nn.Linear(30, 200), torch.nn.Sigmoid(),
        torch.nn.Linear(200, 100), torch.nn.Sigmoid(),
        torch.nn.Linear(100, 1), torch.nn.Sigmoid()).to(device)
error = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the model:
loss_train, loss_test = [], []
acc_train, acc_test = [], []
for ep in range(n_epoch):
    # Forward propagation:
    output = model(data)
    loss = error(output, target)

    loss_train.append(loss.item())
    acc_train.append(score(output, target))

    # Backward propagation:
    loss.backward()

    # Update model parameters:
    optimizer.step()

    # Evaluate the trained model:
    predict = model(x_test)
    loss_test.append(error(predict, y_test).item())
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