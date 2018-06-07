import torch
import matplotlib.pyplot as plt

# Set hyper-parameters:
n_epoch, lr = 100, 1.0E-6

# Load data:
data = torch.tensor([[73, 80, 75], [93, 88, 93], [89, 91, 90],
                     [96, 98, 100], [73, 66, 70]]).float()
target = torch.tensor([[152], [185], [180], [196], [142]]).float()

# Setup a model:
model = torch.nn.Linear(3,1)
error = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train the model:
loss_train = []
for epoch in range(n_epoch):
    optimizer.zero_grad()

    # Forward propagation:
    output = model(data)
    loss = error(output, target)
    loss_train.append(loss)

    # Backward propagation:
    loss.backward()

    # Update model parameters:
    optimizer.step()

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss_train[epoch]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.detach().numpy(), 'x')
ax2.plot(loss_train)