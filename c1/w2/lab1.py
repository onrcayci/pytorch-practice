import numpy as np
from torch import Tensor, float, nn, optim
from torchvision.datasets import FashionMNIST

# load the Fashion MNIST dataset
train_fmnist = FashionMNIST(root=".", download=True, train=True)
test_fmnist = FashionMNIST(root=".", download=True, train=False)
training_images = train_fmnist.data
training_labels = train_fmnist.targets
test_images = test_fmnist.data
test_labels = test_fmnist.targets

# Normalize the pixel values of the training and test images
training_images = training_images / 255
test_images = test_images / 255

# Build the classification model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28 * 28, out_features=28 * 28),
    nn.ReLU(),
    nn.Linear(in_features=28 * 28, out_features=10),
    nn.Softmax(dim=1),
)

print(model)

# Train the model
optimizer = optim.Adam(model.parameters(), 0.005)
loss_fn = nn.CrossEntropyLoss()

for i in range(20):
    print(f"Epoch {i+1}", end=" ")
    predictions: Tensor = model.forward(training_images)
    loss: Tensor = loss_fn(predictions, training_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    accuracy = len(np.nonzero(predictions.argmax(dim=1) == training_labels)) / len(
        training_labels
    )
    print(f"loss: {loss.item()}", f"accuracy: {(accuracy * 100):.2f}%")

# Evaluate the model on unseen data
eval_model = model.eval()

predictions = eval_model(test_images)
accuracy = (
    len(np.nonzero(predictions.argmax(dim=1) == test_labels)) / len(test_labels) * 100
)
print(f"Evaluation accuracy: {accuracy:.3f}%")
