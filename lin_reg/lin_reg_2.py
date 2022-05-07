import torch
import numpy as np

inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
dataset = torch.utils.data.TensorDataset(inputs, targets)
batch_size = 3
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)


def model(x):
    return x @ w.t() + b


def mse_loss(predictions, targets):
    difference = predictions - targets
    return torch.sum(difference * difference)/ difference.numel()


epochs = 500
for i in range(epochs):
    # Iterate through training dataloader
    for x,y in train_loader:
        # Generate Prediction
        preds = model(x)
        # Get the loss and perform backpropagation
        loss = mse_loss(preds, y)
        loss.backward()
        # Let's update the weights
        with torch.no_grad():
            w -= w.grad *1e-6
            b -= b.grad * 1e-6
            # Set the gradients to zero
            w.grad.zero_()
            b.grad.zero_()
print(f"Epoch {i}/{epochs}: Loss: {loss}")
