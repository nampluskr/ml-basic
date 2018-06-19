import numpy as np
import torch
import matplotlib.pyplot as plt

def accuracy(y, t):
    return torch.eq(y>0.5, t>0.5).float().mean()

n_epoch, lr = 1000, 0.01

raw_data = np.genfromtxt("diabetes.csv", delimiter=',')
data = torch.from_numpy(raw_data[:, :-1]).float()
target = torch.from_numpy(raw_data[:, -1]).view(-1,1).float()

model = torch.nn.Sequential(torch.nn.Linear(8, 100),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(100,1),
                            torch.nn.Sigmoid())
error = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

loss_train = []
score_train = []
for n in range(n_epoch):
    optimizer.zero_grad()

    output = model(data)
    loss = error(output, target)
    loss_train.append(loss.item())
    score_train.append(accuracy(output, target).item())

    loss.backward()
    optimizer.step()

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.plot(loss_train)
ax2.plot(score_train)
plt.show()
