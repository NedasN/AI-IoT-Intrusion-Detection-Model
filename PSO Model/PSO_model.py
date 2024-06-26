import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import pyswarms as ps
import torch.nn as nn
import torch.optim as optim
from DataManipulation import DataNormalisation as dn
import numpy as np
from torchmetrics.classification import BinaryAccuracy, ConfusionMatrix, BinaryRecall, BinaryPrecision, BinaryF1Score

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())

# Define the model
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 22, bias=True)  # First hidden layer with 22 neurons
        self.layer2 = nn.Linear(22, 14, bias=True)  # Second hidden layer with 14 neurons
        self.layer3 = nn.Linear(14, 6, bias=True)  # Third hidden layer with 6 neurons
        self.layer4 = nn.Linear(6, 2, bias=True)  # fourth hidden layer with 2 neurons
        self.layer5 = nn.Linear(2, 1, bias=True)    # output layer with 1 neuron

    
    def forward(self, x):
        activation = nn.LeakyReLU(0.1)
        x = activation(self.layer1(x))
        x = activation(self.layer2(x))
        x = activation(self.layer3(x))
        x = activation(self.layer4(x))
        x = self.layer5(x)
        return x

def calculate_dimensions(model):
    dimensions = 0
    for param in model.parameters():
        dimensions += param.numel()
    return dimensions

def f(x):
    n_particles = x.shape[0]
    #print(n_particles)
    losses = []
    with torch.no_grad():
        chunks = sum([torch.numel(param) for param in model.parameters()])
        for i in range(n_particles):
            single_param = list(torch.chunk(torch.tensor(x[i]), int(chunks)))
            for param, target_param in zip(model.parameters(), single_param):
                param.data.copy_(target_param.clone())
            prediction = model(train_tensor)
            prediction.squeeze_()
            #loss = criterion(prediction, target_tensor)
            f1 = BinaryAccuracy().to(device)
            loss = float(f1(prediction, target_tensor))
            #losses.append(loss.item())
            loss = 1 - loss
            losses.append(loss)
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
num_samples = len(train_tensor)

# Generate a random permutation of indices
permutation = torch.randperm(num_samples)

# Shuffle both the dataset and labels using the same permutation
train_tensor = train_tensor[permutation]
target_tensor = target_tensor[permutation]

#Model and optimiser
model = MyNeuralNetwork()
model.to(device)
#'bounds':(-2,2)
#'init_pos':None
options = {'c1':1.2, 'c2': 1, 'w':0.8, 'k':30, 'p':1}
dimensions = calculate_dimensions(model)
lower_bound = np.array([-1] * dimensions)
upper_bound = np.array([1] * dimensions)
bounds = (lower_bound, upper_bound)
#print("Dimensions are:",dimensions)
# topology=ps.backend.topology.Random()
#optimizer = ps.single.GlobalBestPSO(n_particles=700, dimensions=dimensions, options=options, bounds=bounds)
optimizer = ps.single.GeneralOptimizerPSO(n_particles=700, dimensions=dimensions, options=options, topology=ps.backend.topology.Ring(), bounds=bounds)

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

    #calculate recall
    recall = BinaryRecall().to(device)
    calculated_recall = recall(predictions, target_tensor)
    print(f'Recall of best model: {float(calculated_recall)}')

    #calculate precision
    precision = BinaryPrecision().to(device)
    calculated_precision = precision(predictions, target_tensor)
    print(f'Precision of best model: {float(calculated_precision)}')

    #calculate f1 score
    f1 = BinaryF1Score().to(device)
    calculated_f1 = f1(predictions, target_tensor)
    print(f'F1 score of best model: {float(calculated_f1)}')

    #get the confusion matrix
    confusion_matrix = ConfusionMatrix(task = "binary", num_classes = 2).to(device)
    matrix = confusion_matrix(predictions, target_tensor)
    print("ConfusionMatrix:" + str(matrix))

