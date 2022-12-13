from torch import Tensor, float, nn, optim, tensor

# GRADED FUNCTION: house_model
def house_model() -> nn.Sequential:
    ### START CODE HERE

    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms.
    # Hint: Remember to explicitly set the dtype as float
    xs = tensor([[1], [2], [3], [4], [5], [6]], dtype=float)
    ys = tensor([[1], [1.5], [2], [2.5], [3], [3.5]], dtype=float)

    # Define your model (should be a model with 1 linear layer)
    model = nn.Sequential(nn.Linear(in_features=1, out_features=1))

    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss funtion
    optimizer = optim.SGD(model.parameters(), 0.01)
    loss_fn = nn.MSELoss()

    # Train your model for 1000 epochs by feeding the i/o tensors
    for i in range(1000):
        print(f"Epoch {i+1}", end=" ")
        predictions: Tensor = model.forward(xs)
        loss: Tensor = loss_fn(predictions, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"loss: {loss.item()}")

    return model.eval()

# Get your trained model
model = house_model()

new_x = 7
prediction = model(tensor([new_x], dtype=float))
print(prediction.item())
