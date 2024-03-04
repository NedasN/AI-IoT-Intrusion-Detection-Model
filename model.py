import pandas as pd
import torch
import torchswarm
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
        self.layer1 = nn.Linear(30, 70, bias=True)  # First hidden layer with 70 neurons
        self.layer2 = nn.Linear(70, 58, bias=True)  # Second hidden layer with 58 neurons
        self.layer3 = nn.Linear(58, 42, bias=True)  # Third hidden layer with 42 neurons
        self.layer4 = nn.Linear(42, 34, bias=True)  # fourth hidden layer with 34 neurons
        self.layer5 = nn.Linear(34, 21, bias=True)  # sixth hidden layer with 21 neurons
        self.layer6 = nn.Linear(21, 16, bias=True)  # seventh hidden layer with 16 neurons
        self.layer7 = nn.Linear(16, 9, bias= True)  # eighth hidden layer with 9 neurons
        self.layer8 = nn.Linear(9, 5, bias=True)    # ninth hidden layer with 5 neurons
        self.layer9 = nn.Linear(5, 1, bias=True)    # output layer with 1 neuron

    
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
        x = torch.relu(self.layer8(x))
        x = self.layer9(x)
        return torch.sigmoid(x)


#process the data and get the train test split
train_tensor, target_tensor = dn.processData()
#print(train.dtypes)
#print(train.head())
#print(train_tensor.dtype)

#Model and optimiser
model = MyNeuralNetwork()
model.to(device)


#print("Predictions",model(train_tensor))
#print("Target", target_tensor)

train_tensor = train_tensor.to(device)
target_tensor = target_tensor.to(device)
#print(model(train_tensor))

#train the model
for epoch in range(250):
    #ask copilot on how to use torchswarm to train a neural network

    print(f"Epoch {epoch+1}/250")
    #print('Prediction', model(train_tensor))
    #print('Target    ', target_tensor)