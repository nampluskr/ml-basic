import torch
import matplotlib.pyplot as plt

# Set hyper-parameters:
n_epoch, lr = 100, 0.01

# Load data: y = 2 x + 1
data = torch.tensor([[1], [2], [3]]).float()
target = torch.tensor([[3], [5], [7]]).float()

# Setup a model:
weight = torch.tensor([[0.1]])
bias = 0.0

# Train the model:
loss_train = []
for epoch in range(n_epoch):
    # Forward propagation:
    output = torch.mm(data, weight) + bias
    loss = torch.mean((output - target)**2)
    loss_train.append(loss)


    # Backward propagation:
    grad_output = 2*(output - target)/data.size(0)
    grad_weight = torch.mm(data.t(), grad_output)
    grad_bias = torch.sum(grad_output, dim=0)

    # Update model parameters:
    weight -= lr*grad_weight
    bias -= lr*grad_bias

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.numpy())
ax2.plot(loss_train)