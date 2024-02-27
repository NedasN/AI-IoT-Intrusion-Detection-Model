import pandas as pd
import torch
import torchswarm
from torch_pso import ParticleSwarmOptimizer
import torch.nn as nn
import torch.optim as optim
import DataNormalisation as dn

#swap to torchswarm instead of toch_pso

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())

# Define the model
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 45, bias=True)  # First hidden layer with 73 neurons
        self.layer2 = nn.Linear(45, 29, bias=True)  # Second hidden layer with 69 neurons
        self.layer3 = nn.Linear(29, 1, bias=True)  # Third hidden layer with 48 neurons
        #self.layer3 = nn.Linear(69, 48, bias=True)  # Third hidden layer with 48 neurons
        #self.layer4 = nn.Linear(48, 34, bias=True)  # Fourth hidden layer with 34 neurons
        #self.layer5 = nn.Linear(34, 1, bias=True)   # Output layer
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.sigmoid(x)
        return x


#process the data and get the train test split
train_tensor, target_tensor = dn.processData()
#print(train.dtypes)
#print(train.head())
#print(train_tensor.dtype)

#Model and optimiser
model = MyNeuralNetwork()
model.to(device)

optim = ParticleSwarmOptimizer(model.parameters(),
                            inertial_weight=0.7,
                            cognitive_coefficient=0.8,
                            social_coefficient=0.9,
                            num_particles=1000,
                            max_param_value=10000,
                            min_param_value=-10000)

#print("Predictions",model(train_tensor))
#print("Target", target_tensor)
criterion = nn.MSELoss()
train_tensor = train_tensor.to(device)
target_tensor = target_tensor.to(device)
#print(model(train_tensor))

#train the model
for epoch in range(250):
    #ask copilot on how to use torchswarm to train a neural network
    def closure():
        optim.zero_grad()
        output = model(train_tensor)

        return criterion(output, target_tensor)
    optim.step(closure)

    print(f"Epoch {epoch+1}/250")
    #print('Prediction', model(train_tensor))
    #print('Target    ', target_tensor)