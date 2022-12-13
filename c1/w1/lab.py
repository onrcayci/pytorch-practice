from torch import float, nn, optim, tensor, Tensor

# Build a simple Sequential model
model = nn.Sequential(nn.Linear(in_features=1, out_features=1))

# Define model optimizer and loss function
optimizer = optim.SGD(model.parameters(), 0.01)
loss_fn = nn.MSELoss()

# Declare model inputs and outputs for training
xs = tensor([[-1], [0], [1], [2], [3], [4]], dtype=float)
ys = tensor([[-3], [-1], [1], [3], [5], [7]], dtype=float)

# Train the model for 500 epochs
for i in range(500):
    print(f"Epoch {i+1}:", end=" ")
    predictions: Tensor = model.forward(xs)
    loss: Tensor = loss_fn(predictions, ys)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"loss: {loss.item()}")

# Make a prediction
model = model.eval()
print(model(tensor([10], dtype=float)).item())
