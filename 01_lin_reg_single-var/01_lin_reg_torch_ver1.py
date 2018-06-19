import torch
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 100, 0.01

# Data:
data = torch.tensor([[1], [2], [3]], dtype=torch.float)
target = torch.tensor([[3], [5], [7]], dtype=torch.float)

# Model:
weight = torch.tensor([[0.1]])
bias = torch.tensor([0.0])

# Training:
loss_train = []
for n in range(n_epoch):
    output = torch.mm(data, weight) + bias
    loss = torch.mean((output - target)**2)
    loss_train.append(loss.item())

    grad_output = 2*(output - target)/data.size(0)
    grad_weight = torch.mm(data.t(), grad_output)
    grad_bias = torch.sum(grad_output, dim=0)

    weight -= lr*grad_weight
    bias -= lr *grad_bias

# Evaluation:
print(weight, bias)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.numpy(), '--')
ax2.plot(loss_train)
plt.show()
