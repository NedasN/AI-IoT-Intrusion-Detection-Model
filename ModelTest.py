import RandomSampleSelection as rss
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, ConfusionMatrix, BinaryRecall, BinaryPrecision, BinaryF1Score
#from GradModel import MyNeuralNetwork
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 22, bias=True)  # First hidden layer with 70 neurons
        self.layer2 = nn.Linear(22, 14, bias=True)  # Second hidden layer with 58 neurons
        self.layer3 = nn.Linear(14, 6, bias=True)  # Third hidden layer with 42 neurons
        self.layer4 = nn.Linear(6, 2, bias=True)  # fourth hidden 
        self.layer5 = nn.Linear(2, 1, bias=True)    # output layer with 1 neuron

    def forward(self, x):
        activation = nn.LeakyReLU(0.1)
        x = activation(self.layer1(x))
        x = activation(self.layer2(x))
        x = activation(self.layer3(x))
        x = activation(self.layer4(x))
        x = self.layer5(x)
        return x

# Get the random packets form dataset
randomSamples, targets = rss.getRandomSamples()

# Load the model
#model = torch.load('LessTrainedModel.pth')
model = torch.load('LessTrainedModel.pth', map_location=torch.device('cpu'))
model.to(device)
randomSamples = randomSamples.to(device)
targets = targets.to(device)
# Evaluate the model
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    predictions = model(randomSamples)
    predictions.squeeze_()
    predictions = torch.sigmoid(predictions)

    #Calculate overall accuracy of the model
    acc = BinaryAccuracy().to(device)
    calculated_acc = acc(predictions, targets)
    print(f'Accuracy of model: {float(calculated_acc)}')

    #calculate recall
    recall = BinaryRecall().to(device)
    calculated_recall = recall(predictions, targets)
    print(f'Recall of model: {float(calculated_recall)}')

    #calculate precision
    precision = BinaryPrecision().to(device)
    calculated_precision = precision(predictions, targets)
    print(f'Precision of model: {float(calculated_precision)}')

    #calculate f1 score
    f1 = BinaryF1Score().to(device)
    calculated_f1 = f1(predictions, targets)
    print(f'F1 score of model: {float(calculated_f1)}')

    #get the confusion matrix
    confusion_matrix = ConfusionMatrix(task = "binary", num_classes = 2).to(device)
    matrix = confusion_matrix(predictions, targets)
    print("ConfusionMatrix:" + str(matrix))