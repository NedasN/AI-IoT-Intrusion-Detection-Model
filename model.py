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
print("Loaded data")

# Select the features to use
data.drop(["ts", "src_ip", "dst_ip", ""], axis=1, inplace=True)

# Split data into train/test(70/30) split
train, test = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
print("Split the dataset into train and test")

# Drop the labels from train data
train = train.drop(train.columns[-1], axis=1)

train = torch.from_numpy(train.values).float().to(device)
test = torch.from_numpy(test.values).float().to(device)