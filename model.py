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
        self.layer1 = nn.Linear(30, 73, bias=True)  # First hidden layer with 73 neurons
        self.layer2 = nn.Linear(73, 69, bias=True)  # Second hidden layer with 69 neurons
        self.layer3 = nn.Linear(69, 2, bias=True)   # Output layer with 2 neurons

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

#process the data and get the train test split
train, test, targets = dn.processData()
#print(train.dtypes)
#print(train.head())
#print(test.head())
train_tensor = torch.tensor(train.values)
#test = torch.tensor(test.values)
#print(train_tensor.dtype)

#Model and optimiser
model = MyNeuralNetwork()

optim = ParticleSwarmOptimizer(
    model.parameters(),
    inertial_weight=0.5,
    cognitive_coefficient= 0.8,
    social_coefficient=0.7,
    num_particles=1000,
    max_param_value= 10000,
    min_param_value=-10000
)

criterion = torch.nn.MSELoss()

for _ in range(100):
    
    def closure():
        # Clear any grads from before the optimization step, since we will be changing the parameters
        optim.zero_grad()  
        return criterion(model.forward(train_tensor), targets)
    
    optim.step(closure)
    print('Prediciton', model.forward(train_tensor))
    print('Target    ', targets)