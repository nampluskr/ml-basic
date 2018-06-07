import torch
import matplotlib.pyplot as plt

# Set hyper-parameters:
n_epoch, lr = 1000, 0.001

# Load data: y = x^2 + 3
torch.manual_seed(0)
noise = torch.nn.init.normal_(torch.Tensor(1000,1))
data = torch.nn.init.uniform_(torch.Tensor(1000,1), -10, 10)
target = data**2 + 3 + noise

# Setup a model:
w1 = torch.randn(1, 200)
b1 = torch.zeros(200)
w2 = torch.randn(200, 100)
b2 = torch.zeros(100)
w3 = torch.randn(100, 1)
b3 = torch.zeros(1)

# Train the model:
loss = torch.zeros(n_epoch)
for epoch in range(n_epoch):
    # Forward propagation:
    lin1 = torch.mm(data, w1) + b1
    act1 = torch.sigmoid(lin1)
    lin2 = torch.mm(act1, w2) + b2
    act2 = torch.sigmoid(lin2)
    output = torch.mm(act2, w3) + b3
    loss[epoch] = (output - target).pow(2).mean()

    # Backward propagation:
    grad_lin3 = 2*(output - target)/data.size(0)
    grad_w3 = torch.mm(act2.t(), grad_lin3)
    grad_b3 = torch.sum(grad_lin3, dim=0)

    grad_act2 = torch.mm(grad_lin3, w3.t())
    grad_lin2 = act2*(1-act2)*grad_act2
    grad_w2 = torch.mm(act1.t(), grad_lin2)
    grad_b2 = torch.sum(grad_lin2, dim=0)

    grad_act1 = torch.mm(grad_lin2, w2.t())
    grad_lin1 = act1*(1-act1)*grad_act1
    grad_w1 = torch.mm(data.t(), grad_lin1)
    grad_b1 = torch.sum(grad_lin1, dim=0)

    # Update model parameters:
    w3 -= lr*grad_w3
    b3 -= lr*grad_b3
    w2 -= lr*grad_w2
    b2 -= lr*grad_b2
    w1 -= lr*grad_w1
    b1 -= lr*grad_b1

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss[epoch]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.numpy(), '.')
ax2.plot(loss.numpy())