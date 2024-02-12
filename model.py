import pandas as pd
import torch
from torch_pso import ParticleSwarmOptimizer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = pd.read_csv('Dataset/TON_IoT/Train_Test_Network.csv')

# Select the features to use

# Split data into train/test(70/30) split
train, test = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)

# Drop the labels from train data
train = train.drop(train.columns[-1], axis=1)

