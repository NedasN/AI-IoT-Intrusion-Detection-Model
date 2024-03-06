import pandas as pd
import torch
import pyswarms as ps
import torch.nn as nn
import torch.optim as optim
import DataNormalisation as dn
import numpy as np
#swap to torchswarm instead of toch_pso

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())

# Define the model
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 24, bias=True)  # First hidden layer with 70 neurons
        self.layer2 = nn.Linear(24, 19, bias=True)  # Second hidden layer with 58 neurons
        self.layer3 = nn.Linear(19, 14, bias=True)  # Third hidden layer with 42 neurons
        self.layer4 = nn.Linear(14, 10, bias=True)  # fourth hidden layer with 34 neurons
        self.layer5 = nn.Linear(10, 6, bias=True)  # sixth hidden layer with 21 neurons
        self.layer6 = nn.Linear(6, 2, bias=True)  # seventh hidden layer with 16 neurons
        self.layer8 = nn.Linear(2, 1, bias=True)    # output layer with 1 neuron

    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        x = torch.relu(x)
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.relu(self.layer6(x))
        x = self.layer8(x)
        return x

def calculate_dimensions(model):
    dimensions = 0
    for param in model.parameters():
        dimensions += param.numel()
    return dimensions

def f(x):
    n_particles = x.shape[0]
    losses = []
    chunks = sum([torch.numel(param) for param in model.parameters()])
    for i in range(n_particles):
        single_param = list(torch.chunk(torch.tensor(x[i]), int(chunks)))
        for param, target_param in zip(model.parameters(), single_param):
            param.data.copy_(target_param.clone())
        prediction = model(train_tensor)
        prediction.squeeze_()
        loss = criterion(prediction, target_tensor)
        losses.append(loss.item())
    return np.array(losses)

#process the data and get the train test split
train_tensor, target_tensor = dn.processData()
#print(train.dtypes)
#print(train.head())
#print(train_tensor.dtype)

#Model and optimiser
model = MyNeuralNetwork()
model.to(device)
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
dimensions = calculate_dimensions(model)
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)
#print("Predictions",model(train_tensor))
#print("Target", target_tensor)

train_tensor = train_tensor.to(device)
target_tensor = target_tensor.float().to(device)
#print(model(train_tensor))
criterion = torch.nn.BCEWithLogitsLoss()
cost, pos = optimizer.optimize(f, iters=1000, verbose=3)
#train the model
'''for epoch in range(250):
    #ask copilot on how to use torchswarm to train a neural network

    print(f"Epoch {epoch+1}/250")
    #print('Prediction', model(train_tensor))
    #print('Target    ', target_tensor)'''