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
model = torch.nn.Sequential(
            torch.nn.Linear(1,200),
            torch.nn.Sigmoid(),
            torch.nn.Linear(200,100),
            torch.nn.Sigmoid(),
            torch.nn.Linear(100,1))
error = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train the model:
loss_train = torch.zeros(n_epoch)
for epoch in range(n_epoch):
    optimizer.zero_grad()

    # Forward propagation:
    output = model(data)
    loss = error(output, target)
    loss_train[epoch] = loss.item()

    # Backward propagation:
    loss.backward()

    # Update model parameters:
    optimizer.step()

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss_train[epoch]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.detach().numpy(), '.')
ax2.plot(loss_train.numpy())