import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def processData():
    # Load data
    data = pd.read_csv('Dataset/TON_IoT/Train_Test_Network.csv')
    #print("Loaded data")

    # Select the features to use
    data.drop(["ts", "src_ip", "dst_ip", "http_user_agent", "http_orig_mime_types", "ssl_version", "ssl_cipher", "ssl_subject", "ssl_issuer", "http_uri", "http_version", 'http_user_agent', "http_orig_mime_types", "http_resp_mime_types", "weird_notice", "label"], axis=1, inplace=True)

    #convert the string types of data into numeric categories
    #data = dn.StringsToCategories(data)
    protocol_map = {'tcp': 1, 'udp': 2, 'icmp': 3}
    service_map = {'-': 0, 'http': 1, 'dns' : 2, 'ftp' : 3, 'smb' : 4, 'ssl' : 5, 'dhcp' : 6, 'gssapi' : 7, 'smb;gssapi' : 8,'dce_rpc' : 9}
    conn_state_map = {'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4, 'RSTRH': 5, 'S0': 6, 'S1': 7, 'S2': 8, 'S3': 9, 'SF': 10, 'SH' : 11, 'SHR' : 12}
    data['proto'] = data['proto'].map(protocol_map)
    data['service'] = data['service'].map(service_map)
    data['conn_state'] = data['conn_state'].map(conn_state_map)
    #potentially replace with matching with knonw attack dns adresses
    #currently just anwsers true or false if it acessed a dns server
    data['dns_query'] = data['dns_query'].replace({'-': 0, '.*': 1}, regex=True)
    #potentially needs 3 classes to represent the values that are not applicable aka - values
    data[['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_resumed', 'ssl_established']] = data[['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_resumed', 'ssl_established']].replace({'-|F': 0, 'T': 1}, regex=True)
    data['http_trans_depth'] = data['http_trans_depth'].replace({'-': 0}, regex=True)
    data['http_trans_depth'] = data['http_trans_depth'].astype(int)
    data['http_method'] = data['http_method'].replace({'-': 0, 'GET': 1, 'HEAD': 2, 'POST': 3}, regex=True)
    data['weird_name'] = data['weird_name'].replace({'-': 0, '.*': 1}, regex=True)
    data['weird_addl'] = data['weird_addl'].replace({'-': 0}, regex=True)
    data['weird_addl'] = data['weird_addl'].astype(int)
    data['type'] = data['type'].replace({'normal': 0, 'scanning': 1, 'dos': 2, 'ddos': 3, 'backdoor': 4, 'injection': 5, 'mitm': 6, 'password': 7, 'ransomware': 8, 'xss': 9}, regex=True)


    #print(data['weird_addl'].unique())
    #print(data['conn_state'].unique())
    #print(data[data.columns[0]].count())
    data.drop(data[(data['duration'] > 12000)].index, inplace=True)
    #list_to_standardise = ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes', 'src_pkts', 'dst_pkts', 'src_ip_bytes', 'dst_ip_bytes']

    list_to_normalise = ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes', 'src_pkts', 'dst_pkts', 'src_ip_bytes', 'dst_ip_bytes']
    for column in list_to_normalise:
        data[column] = np.log1p(data[column])
        #data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    #print(data.head(5))
    #print(len(data.index))


    #Get the tensor of the data as well as the tensor containing target values
    #duration_tensor = torch.tensor(data['duration'].values)
    #num_bins = int(duration_tensor.size(0) ** 0.5)
    #duration_discretized = torch.bucketize(duration_tensor, boundaries=torch.linspace(duration_tensor.min(), duration_tensor.max(), steps=num_bins))
    target_tensor = torch.tensor(data['type'].values)
    data.drop(['type'], axis=1, inplace=True)
    #data.drop(['duration'], axis=1, inplace=True)
    train_tensor = torch.tensor(data.values, requires_grad=False).float()
    #train_tensor = torch.cat((train_tensor, duration_discretized.unsqueeze(1)), 1)
    #print(train_tensor.dtype)
    #print(target_tensor.dtype)
    return train_tensor, target_tensor


#_,tensor = processData()