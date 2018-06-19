import torch
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 100, 1.0E-6

# Data:
data = torch.tensor([[73, 80, 75], [93, 88, 93], [89, 91, 90],
                     [96, 98, 100], [73, 66, 70]]).float()
target = torch.tensor([[152], [185], [180], [196], [142]]).float()

# Model:
model = torch.nn.Linear(3,1)
error = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training:
loss_train = []
for epoch in range(n_epoch):
    optimizer.zero_grad()

    output = model(data)
    loss = error(output, target)
    loss_train.append(loss)

    loss.backward()
    optimizer.step()

# Evaluation:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss_train[epoch]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
ax1.plot(data.numpy(), target.numpy(), 'o')
ax1.plot(data.numpy(), output.detach().numpy(), 'x')
ax2.plot(loss_train)