import pandas as pd
import torch
import pyswarms as ps
import torch.nn as nn
import torch.optim as optim
import DataNormalisation as dn
import numpy as np
from torchmetrics.classification import BinaryAccuracy
#swap to pyswarm instead of toch_pso
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())

# Define the model
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 22, bias=True)  # First hidden layer with 70 neurons
        self.layer2 = nn.Linear(22, 14, bias=True)  # Second hidden layer with 58 neurons
        self.layer3 = nn.Linear(14, 6, bias=True)  # Third hidden layer with 42 neurons
        self.layer4 = nn.Linear(6, 2, bias=True)  # fourth hidden layer with 34 neurons
        #self.layer5 = nn.Linear(10, 6, bias=True)  # sixth hidden layer with 21 neurons
        #self.layer6 = nn.Linear(6, 2, bias=True)  # seventh hidden layer with 16 neurons
        self.layer8 = nn.Linear(2, 1, bias=True)    # output layer with 1 neuron

    
    def forward(self, x):
        activation = nn.LeakyReLU(0.1)
        x = activation(self.layer1(x))
        x = activation(self.layer2(x))
        x = activation(self.layer3(x))
        x = activation(self.layer4(x))
        #x = torch.relu(self.layer5(x))
        #x = torch.relu(self.layer6(x))
        x = self.layer8(x)
        return x

def calculate_dimensions(model):
    dimensions = 0
    for param in model.parameters():
        dimensions += param.numel()
    return dimensions

def f(x):
    n_particles = x.shape[0]
    print(n_particles)
    losses = []
    with torch.no_grad():
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

def reshape_parameters(flattened_params, model):
    reshaped_params = []
    current_index = 0

    # Iterate through the layers of the model
    for param in model.parameters():
        param_size = torch.prod(torch.tensor(param.shape)).item()
        # Extract a chunk from flattened_params based on the size of the current layer's parameters
        chunk = flattened_params[current_index:current_index + param_size]
        # Convert the chunk to a tensor and reshape to match the shape of the current layer's parameters
        reshaped_chunk = torch.tensor(chunk, dtype=param.dtype).view(param.shape)
        # Append the reshaped chunk to the list of reshaped parameters
        reshaped_params.append(reshaped_chunk)
        # Update the current index for the next layer
        current_index += param_size

    return reshaped_params

#process the data and get the train test split
train_tensor, target_tensor = dn.processData()
#print(train.dtypes)
#print(train.head())
#print(train_tensor.dtype)

#Model and optimiser
model = MyNeuralNetwork()
model.to(device)
#'bounds':(-2,2)
options = {'c1':0.7, 'c2': 0.3, 'w':0.6, 'k':7, 'p':2,'init_pos':None}
dimensions = calculate_dimensions(model)
#print("Dimensions are:",dimensions)
optimizer = ps.single.GeneralOptimizerPSO(n_particles=50, dimensions=dimensions, options=options, topology=ps.backend.topology.Ring())
#print("Predictions",model(train_tensor))
#print("Target", target_tensor)

train_tensor = train_tensor.to(device)
target_tensor = target_tensor.float().to(device)
#print(model(train_tensor))
criterion = torch.nn.BCEWithLogitsLoss()
cost, pos = optimizer.optimize(f, iters=300, verbose=3)
# After training the model
#set the model parameters to the best found
reshaped_params = reshape_parameters(pos, model)
with torch.no_grad():
    for param, target_param in zip(model.parameters(), reshaped_params):
        param.data.copy_(target_param.clone())

with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    predictions = model(train_tensor)
    predictions.squeeze_()
    predictions = torch.sigmoid(predictions)
    #predictions = (predictions > 0.5).float()
    acc = BinaryAccuracy().to(device)
    calculated_acc = acc(predictions, target_tensor)
    print(f'Accuracy of best model: {float(calculated_acc)}')
