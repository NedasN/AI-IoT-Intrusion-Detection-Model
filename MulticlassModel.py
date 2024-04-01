import torch
import torch.nn as nn
import torch.optim as optim
import DataLoadForMulticlass as dn
import MulticlassDataNormalisation as wholeData
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, ConfusionMatrix, MulticlassRecall, MulticlassPrecision, MulticlassF1Score
#swap to pyswarm instead of toch_pso
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())

# Define the model
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 22, bias=True)  # First hidden layer with 70 neurons
        self.layer2 = nn.Linear(22, 16, bias=True)  # Second hidden layer with 58 neurons
        self.layer3 = nn.Linear(16, 14, bias=True)  # Third hidden layer with 42 neurons
        self.layer4 = nn.Linear(14, 12, bias=True)  # fourth hidden 
        self.layer5 = nn.Linear(12, 10, bias=True)    # output layer with 10 neuron

    
    def forward(self, x):
        activation = nn.LeakyReLU(0.1)
        output_activation = nn.LogSoftmax(dim=1)
        x = activation(self.layer1(x))
        x = activation(self.layer2(x))
        x = activation(self.layer3(x))
        x = activation(self.layer4(x))
        #x = torch.relu(self.layer5(x))
        #x = torch.relu(self.layer6(x))
        x = self.layer5(x)
        return output_activation(x)


#process the data and get the train test split
train_tensor, train_target_tensor, test_tensor, test_target_tensor = dn.processData()
#print(train.dtypes)
#print(train.head())
#print(train_tensor.dtype)

#Model and optimiser
model = MyNeuralNetwork()
model.to(device)

num_epochs = 700
params = {
    'batch_size': 512,
    }


train_dataloader = torch.utils.data.DataLoader(list(zip(train_tensor, train_target_tensor)), **params)
test_dataloader = torch.utils.data.DataLoader(list(zip(test_tensor, test_target_tensor)), **params)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#train the model
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs) # Reshape inputs if needed
        #outputs.squeeze_()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader):.3f}')
    running_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #outputs.squeeze_()
            loss = loss_fn(outputs, labels)

            # Calculate accuracy
            acc = MulticlassAccuracy(num_classes=10).to(device)
            calculated_acc = acc(outputs, labels)
        print(f'Epoch {epoch + 1}, Validation Loss: {loss:.3f}, Validation Accuracy: {float(calculated_acc):.3f}')

    #print(f'Epoch {epoch + 1}, Validation Accuracy: {float(calculated_acc):.3f}')

print('Finished Training')
print('Starting Evaluation')
# After training the model
#find the accuracy for the best model and whole dataset
#get the confusion matrix
full_tensor, full_labels = wholeData.processData()
full_tensor, full_labels = full_tensor.to(device), full_labels.to(device)
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    predictions = model(full_tensor)
    #predictions.squeeze_()

    #Calculate overall accuracy of the model
    acc = MulticlassAccuracy(num_classes=10).to(device)
    calculated_acc = acc(predictions, full_labels)
    print(f'Accuracy of best model: {float(calculated_acc)}')

    #calculate recall
    recall = MulticlassRecall(num_classes=10).to(device)
    calculated_recall = recall(predictions, full_labels)
    print(f'Recall of best model: {float(calculated_recall)}')

    #calculate precision
    precision = MulticlassPrecision(num_classes=10).to(device)
    calculated_precision = precision(predictions, full_labels)
    print(f'Precision of best model: {float(calculated_precision)}')

    #calculate f1 score
    f1 = MulticlassF1Score(num_classes=10).to(device)
    calculated_f1 = f1(predictions, full_labels)
    print(f'F1 score of best model: {float(calculated_f1)}')

    #get the confusion matrix
    #predictions.squeeze_()
    confusion_matrix = ConfusionMatrix(task = "multiclass", num_classes = 10).to(device)
    matrix = confusion_matrix(predictions, full_labels)
    print("ConfusionMatrix:" + str(matrix))

print('Finished Evaluation')
'''
print('Saving the model')
file_path = 'MulticlassModel.pth'

# Save the entire model
torch.save(model, file_path)

print('Done')'''