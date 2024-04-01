import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Dataset/TON_IoT/Train_Test_Network.csv')

count = data['type'].value_counts()

count.plot(kind='bar')

plt.title('Distribution of the labels')
plt.show()