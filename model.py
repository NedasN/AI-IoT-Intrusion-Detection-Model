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
        self.layer3 = nn.Linear(69, 1, bias=True)  # Third hidden layer with 48 neurons
        #self.layer3 = nn.Linear(69, 48, bias=True)  # Third hidden layer with 48 neurons
        #self.layer4 = nn.Linear(48, 34, bias=True)  # Fourth hidden layer with 34 neurons
        #self.layer5 = nn.Linear(34, 1, bias=True)   # Output layer
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x


#process the data and get the train test split
train_tensor, target_tensor = dn.processData()
#print(train.dtypes)
#print(train.head())
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

#print("Predictions",model(train_tensor))
#print("Target", target_tensor)
criterion = nn.CrossEntropyLoss()

#print(model(train_tensor).size())

print("Got to before the loop")
for epoch in range(500):
    print("in the loop now")
    def closure():
        # Clear any grads from before the optimization step, since we will be changing the parameters
        optim.zero_grad()
        out = model(train_tensor)
        #out = out.view(-1)
        print("in Closure")
        return criterion(out, target_tensor)
    
    optim.step(closure)
    print(f"Epoch {epoch}/500")
    #print('Prediction', model(train_tensor))
    #print('Target    ', target_tensor)