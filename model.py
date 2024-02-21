import pandas as pd
import torch
from torch_pso import ParticleSwarmOptimizer
import torch.nn as nn
import torch.optim as optim
import DataNormalisation as dn

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 73)  # First hidden layer with 73 neurons
        self.layer2 = nn.Linear(73, 69)  # Second hidden layer with 69 neurons
        self.layer3 = nn.Linear(69, 2)   # Output layer with 2 neurons

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

#process the data and get the train test split
train, test = dn.processData()

print(train.head())
print(test.head())
train_tensor = torch.tensor(train.values)
test = torch.tensor(test.values)
