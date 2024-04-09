import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

data = pd.read_csv('Dataset/TON_IoT/Train_Test_Network.csv')

count = data['label'].value_counts()

count.plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title('Packet Balance in the Dataset')
plt.show()