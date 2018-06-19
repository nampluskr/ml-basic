import torch
import matplotlib.pyplot as plt

use_gpu = 0
device = torch.device('cuda') if use_gpu else torch.device('cpu')

# Hyper-parameters:
n_epoch, lr = 100, 0.01

# Data:
data = torch.tensor([[1], [2], [3]], dtype=torch.float).to(device)
target = torch.tensor([[3], [5], [7]], dtype=torch.float).to(device)

# Model:
model = torch.nn.Linear(1, 1).to(device)
error = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training:
loss_train = []
for n in range(n_epoch):
    optimizer.zero_grad()

    output = model(data)
    loss = error(output, target)
    loss_train.append(loss.item())

    loss.backward()
    optimizer.step()

# Evaluation:
if use_gpu:
    data = data.cpu()
    target = target.cpu()

print(list(model.parameters()))
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.detach().numpy(), '--')
ax2.plot(loss_train)
plt.show()
