import pandas as pd
import torch
#import torchswarm
import copy
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
        self.layer1 = nn.Linear(30, 30, bias=True)  # First hidden layer with 70 neurons
        self.layer2 = nn.Linear(30, 25, bias=True)  # Second hidden layer with 58 neurons
        self.layer3 = nn.Linear(25, 22, bias=True)  # Third hidden layer with 42 neurons
        self.layer4 = nn.Linear(22, 18, bias=True)  # fourth hidden layer with 34 neurons
        self.layer5 = nn.Linear(18, 13, bias=True)  # sixth hidden layer with 21 neurons
        self.layer6 = nn.Linear(13, 9, bias=True)  # seventh hidden layer with 16 neurons
        self.layer7 = nn.Linear(9, 5, bias=True)    # ninth hidden layer with 5 neurons
        self.layer8 = nn.Linear(5, 1, bias=True)    # output layer with 1 neuron

    
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
        x = torch.relu(self.layer7(x))
        x = self.layer8(x)
        return torch.sigmoid(x)


#process the data and get the train test split
train_tensor, target_tensor = dn.processData()
#print(train.dtypes)
#print(train.head())
#print(train_tensor.dtype)

#Model and optimiser
model = MyNeuralNetwork()
model = model.to(device)

optim = ParticleSwarmOptimizer(model.parameters(),
                            inertial_weight=0.9,
                            cognitive_coefficient=1.8,
                            social_coefficient=2,
                            num_particles=500,
                            max_param_value=1000,
                            min_param_value=-1000)

#print("Predictions",model(train_tensor))
#print("Target", target_tensor)
#train_tensor = train_tensor.to(device)
#target_tensor = target_tensor.to(device)
#print(model(train_tensor))

criterion = nn.MSELoss()
best_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())

params = {
    'batch_size': 2046,
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
            loss = criterion(output, target_batch)

            # Save if this is the best model we've seen so far
            if loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

        return loss
    
    optim.step(closure)

    print(f"Epoch {epoch+1}/350, Loss: {best_loss}")

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

'''
for epoch in range(250):
    def closure():
        optim.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, target_tensor)

        # Save if this is the best model we've seen so far
        if loss < best_loss:
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())

        return loss
    
    optim.step(closure)

    print(f"Epoch {epoch+1}/250")
'''