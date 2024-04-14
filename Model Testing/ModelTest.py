import RandomSampleSelection as rss
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DataManipulation import DataNormalisation as BinaryWholeData
from DataManipulation import MulticlassDataNormalisation as MulticlassWholeData
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, ConfusionMatrix, BinaryRecall, BinaryPrecision, BinaryF1Score, MulticlassAccuracy, ConfusionMatrix, MulticlassRecall, MulticlassPrecision, MulticlassF1Score
#from GradModel import MyNeuralNetwork
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 22, bias=True)  # First hidden layer with 22 neurons
        self.layer2 = nn.Linear(22, 14, bias=True)  # Second hidden layer with 14 neurons
        self.layer3 = nn.Linear(14, 6, bias=True)  # Third hidden layer with 6 neurons
        self.layer4 = nn.Linear(6, 2, bias=True)  # fourth hidden
        self.layer5 = nn.Linear(2, 1, bias=True)  # output layer with 1 neuron

    def forward(self, x):
        activation = nn.LeakyReLU(0.1)
        x = activation(self.layer1(x))
        x = activation(self.layer2(x))
        x = activation(self.layer3(x))
        x = activation(self.layer4(x))
        x = self.layer5(x)
        return x

class MulticlassNeuralNetwork(nn.Module):
    def __init__(self):
        super(MulticlassNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 22, bias=True)  # First hidden layer with 22 neurons
        self.layer2 = nn.Linear(22, 16, bias=True)  # Second hidden layer with 16 neurons
        self.layer3 = nn.Linear(16, 14, bias=True)  # Third hidden layer with 14 neurons
        self.layer4 = nn.Linear(14, 12, bias=True)  # fourth hidden 
        self.layer5 = nn.Linear(12, 10, bias=True)  # output layer with 10 neuron

    def forward(self, x):
        activation = nn.LeakyReLU(0.1)
        output_activation = nn.LogSoftmax(dim=1)
        x = activation(self.layer1(x))
        x = activation(self.layer2(x))
        x = activation(self.layer3(x))
        x = activation(self.layer4(x))
        x = self.layer5(x)
        return output_activation(x)

def randomSubSet(task):
    # Get the random packets form dataset
    randomSamples, targets = rss.getRandomSamples(task)

    # Load the model
    #use this if you do not have gpu supper enabled
    #model = torch.load('Gradient Descent Model/GradTrainedModel.pth', map_location=torch.device('cpu'))
    if task == 'binary':
        model = MyNeuralNetwork()
        model_weigts = torch.load('Gradient Descent Model/BinaryGradWeights.pt')
        model.load_state_dict(model_weigts)

    elif task == 'multiclass':
        model = MulticlassNeuralNetwork()
        model_weigts = torch.load('Gradient Descent Model/MulticlassModelWeights.pth')
        model.load_state_dict(model_weigts)
    model.to(device)
    randomSamples = randomSamples.to(device)
    targets = targets.to(device)
    return model, randomSamples, targets

def WholeDataset(task):
    if task == 'binary':
        data, targets = BinaryWholeData.processData()
        # Load the model
        model = MyNeuralNetwork()
        model_weigts = torch.load('Gradient Descent Model/BinaryGradWeights.pt')
        model.load_state_dict(model_weigts)
    elif task == 'multiclass':
        model = MulticlassNeuralNetwork()
        model_weigts = torch.load('Gradient Descent Model/MulticlassModelWeights.pth')
        model.load_state_dict(model_weigts)
        data, targets = MulticlassWholeData.processData()

    model.to(device)
    data = data.to(device)
    targets = targets.to(device)
    return model, data, targets

def evaluateBinaryModel(set):
    if set == 'sub':
        model, data, targets = randomSubSet('binary')
    elif set == 'all':
        model, data, targets = WholeDataset('binary')
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predictions = model(data)
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

def evaluateMulticlassModel(set):
    if set == 'sub':
        model, data, targets = randomSubSet('multiclass')
    elif set == 'all':
        model, data, targets = WholeDataset('multiclass')
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predictions = model(data)

        #Calculate overall accuracy of the model
        acc = MulticlassAccuracy(num_classes=10).to(device)
        calculated_acc = acc(predictions, targets)
        print(f'Accuracy of best model: {float(calculated_acc)}')

        #calculate recall
        recall = MulticlassRecall(num_classes=10).to(device)
        calculated_recall = recall(predictions, targets)
        print(f'Recall of best model: {float(calculated_recall)}')

        #calculate precision
        precision = MulticlassPrecision(num_classes=10).to(device)
        calculated_precision = precision(predictions, targets)
        print(f'Precision of best model: {float(calculated_precision)}')

        #calculate f1 score
        f1 = MulticlassF1Score(num_classes=10).to(device)
        calculated_f1 = f1(predictions, targets)
        print(f'F1 score of best model: {float(calculated_f1)}')

        #get the confusion matrix
        #predictions.squeeze_()
        confusion_matrix = ConfusionMatrix(task = "multiclass", num_classes = 10).to(device)
        matrix = confusion_matrix(predictions, targets)
        print("ConfusionMatrix:" + str(matrix))

# Evaluate the models
#evaluateBinaryModel('sub')
#evaluateBinaryModel('all')

evaluateMulticlassModel('sub')
#evaluateMulticlassModel('all')