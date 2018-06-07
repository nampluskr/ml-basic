import torch
import matplotlib.pyplot as plt

# Set hyper-parameters:
n_epoch, lr = 100, 1.0E-6

# Load data:
data = torch.tensor([[73, 80, 75], [93, 88, 93], [89, 91, 90],
                     [96, 98, 100], [73, 66, 70]]).float()
target = torch.tensor([[152], [185], [180], [196], [142]]).float()

# Setup a model:
weight = torch.rand(3,1)
bias = torch.zeros(1)

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
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss_train[epoch]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.numpy(), 'x')
ax2.plot(loss_train)