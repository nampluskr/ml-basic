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
w1 = torch.randn(1,200, requires_grad=True)
b1 = torch.zeros(200, requires_grad=True)
w2 = torch.randn(200,100, requires_grad=True)
b2 = torch.zeros(100, requires_grad=True)
w3 = torch.randn(100,1, requires_grad=True)
b3 = torch.zeros(1, requires_grad=True)

# Train the model:
loss_train = torch.zeros(n_epoch)
for epoch in range(n_epoch):
    # Forward propagation:
    lin1 = torch.mm(data, w1) + b1
    act1 = torch.sigmoid(lin1)
    lin2 = torch.mm(act1, w2) + b2
    act2 = torch.sigmoid(lin2)
    output = torch.mm(act2, w3) + b3
    loss = (output - target).pow(2).mean()
    loss_train[epoch] = loss.item()

    # Backward propagation:
    loss.backward()

    # Update model parameters:
    w3.data -= lr*w3.grad.data
    b3.data -= lr*b3.grad.data
    w2.data -= lr*w2.grad.data
    b2.data -= lr*b2.grad.data
    w1.data -= lr*w1.grad.data
    b1.data -= lr*b1.grad.data

    w3.grad.zero_()
    b3.grad.zero_()
    w2.grad.zero_()
    b2.grad.zero_()
    w1.grad.zero_()
    b1.grad.zero_()

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss_train[epoch]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.detach().numpy(), '.')
ax2.plot(loss_train.numpy())