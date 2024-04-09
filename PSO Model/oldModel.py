import pandas as pd
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy
from torch_pso import ParticleSwarmOptimizer
import torch.nn as nn
import torch.optim as optim
from DataManipulation import DataNormalisation as dn

#swap to torchswarm instead of toch_pso

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
        return torch.sigmoid(x)


#process the data and get the train test split
train_tensor, target_tensor = dn.processData()

#Model and optimiser
model = MyNeuralNetwork()
model = model.to(device)

optim = ParticleSwarmOptimizer(model.parameters(),
                            inertial_weight=0.9,
                            cognitive_coefficient=1.8,
                            social_coefficient=2,
                            num_particles=200,
                            max_param_value=10,
                            min_param_value=-10)

criterion = nn.BCELoss()
best_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

params = {
    'batch_size': 512,
    'shuffle': True,
    }

dataloader_train = iter(torch.utils.data.DataLoader(train_tensor, **params))
dataloader_target = iter(torch.utils.data.DataLoader(target_tensor, **params))

#train the model

print("Started Training")
for epoch in range(700):
    try:
        train_batch = next(dataloader_train)
        target_batch = next(dataloader_target)
        train_batch = train_batch.to(device)
        target_batch = target_batch.to(device)
    except StopIteration:
        #If iterator runs out of batches we need to reset it to the start 
        dataloader_train = iter(torch.utils.data.DataLoader(train_tensor, **params))
        dataloader_target = iter(torch.utils.data.DataLoader(target_tensor, **params))
        train_batch = next(dataloader_train)
        target_batch = next(dataloader_target)
        train_batch = train_batch.to(device)
        target_batch = target_batch.to(device)

    def closure():
        optim.zero_grad()
        global best_loss
        global best_model_wts
        with torch.no_grad():
            output = model(train_batch)
            output.squeeze_()
            loss = criterion(output, target_batch)

            # Save if this is the best model we've seen so far
            if loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

        return loss
    
    optim.step(closure)

    print(f"Epoch {epoch+1}/700, Loss: {best_loss}")

print("Finished Training...")
print("Evaluating best model...")
model.load_state_dict(best_model_wts)
train_tensor = train_tensor.to(device)
target_tensor = target_tensor.to(device)
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    predictions = model(train_tensor)
    _, predicted_labels = torch.max(predictions, 1)
    correct_predictions = (predicted_labels == target_tensor).sum().item()
    total_samples = target_tensor.size(0)
    accuracy = correct_predictions / total_samples * 100.0
    print(f"Accuracy of the best model: {accuracy:.2f}%")
