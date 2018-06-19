import torch
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 100, 0.01

# Data:
data = torch.tensor([[1], [2], [3]], dtype=torch.float)
target = torch.tensor([[3], [5], [7]], dtype=torch.float)

# Model:
weight = torch.tensor([[0.1]], requires_grad=True)
bias = torch.tensor([0.0], requires_grad=True)

# Training:
loss_train = []
for n in range(n_epoch):
    output = torch.mm(data, weight) + bias
    loss = torch.mean((output - target)**2)
    loss_train.append(loss.item())

    loss.backward()

    weight.data -= lr*weight.grad.data
    bias.data -= lr *bias.grad.data

    weight.grad.zero_()
    bias.grad.zero_()

# Evaluation:
print(weight, bias)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.detach().numpy(), '--')
ax2.plot(loss_train)
plt.show()
