import torch

# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5])

# Perform some operations on the tensor
y = x + 2
z = torch.sin(y)

# Print the results
print("Original tensor:", x)
print("Modified tensor:", y)
print("Sinusoidal tensor:", z)
import torch.nn as nn

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(5, 1)  # Fully connected layer with input size 5 and output size 1

    def forward(self, x):
        x = self.fc(x)
        return x

# Create an instance of the network
net = Net()

# Print the weights and biases of the network
for name, param in net.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

# Perform forward pass through the network
output = net(x)

# Print the results
print("Original tensor:", x)
print("Output tensor:", output)
