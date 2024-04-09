import pandas as pd
import random
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def getRandomSamples():
    p = 0.01  # 4000-sh lines from our dataset
    # keep the header, then take only 1% of lines
    # if random from [0,1] interval is greater than 0.01 the row will be skipped
    df = pd.read_csv(
            'Dataset/TON_IoT/Train_Test_Network.csv',
            header=0, 
            skiprows=lambda i: i>0 and random.random() > p
    )
    
    df.drop(["ts", "src_ip", "dst_ip", "http_user_agent", "http_orig_mime_types", "ssl_version", "ssl_cipher", "ssl_subject", "ssl_issuer", "http_uri", "http_version", 'http_user_agent', "http_orig_mime_types", "http_resp_mime_types", "weird_notice", "type"], axis=1, inplace=True)
    #convert the string types of data into numeric categories
    #data = dn.StringsToCategories(data)
    protocol_map = {'tcp': 1, 'udp': 2, 'icmp': 3}
    service_map = {'-': 0, 'http': 1, 'dns' : 2, 'ftp' : 3, 'smb' : 4, 'ssl' : 5, 'dhcp' : 6, 'gssapi' : 7, 'smb;gssapi' : 8,'dce_rpc' : 9}
    conn_state_map = {'OTH': 0, 'REJ': 1, 'RSTO': 2, 'RSTOS0': 3, 'RSTR': 4, 'RSTRH': 5, 'S0': 6, 'S1': 7, 'S2': 8, 'S3': 9, 'SF': 10, 'SH' : 11, 'SHR' : 12}
    df['proto'] = df['proto'].map(protocol_map)
    df['service'] = df['service'].map(service_map)
    df['conn_state'] = df['conn_state'].map(conn_state_map)
    #potentially replace with matching with knonw attack dns adresses
    #currently just anwsers true or false if it acessed a dns server
    df['dns_query'] = df['dns_query'].replace({'-': 0, '.*': 1}, regex=True)
    #potentially needs 3 classes to represent the values that are not applicable aka - values
    df[['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_resumed', 'ssl_established']] = df[['dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_resumed', 'ssl_established']].replace({'-|F': 0, 'T': 1}, regex=True)
    df['http_trans_depth'] = df['http_trans_depth'].replace({'-': 0}, regex=True)
    df['http_trans_depth'] = df['http_trans_depth'].astype(int)
    df['http_method'] = df['http_method'].replace({'-': 0, 'GET': 1, 'HEAD': 2, 'POST': 3}, regex=True)
    df['weird_name'] = df['weird_name'].replace({'-': 0, '.*': 1}, regex=True)
    df['weird_addl'] = df['weird_addl'].replace({'-': 0}, regex=True)
    df['weird_addl'] = df['weird_addl'].astype(int)

    list_to_normalise = ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes', 'src_pkts', 'dst_pkts', 'src_ip_bytes', 'dst_ip_bytes']
    for column in list_to_normalise:
        df[column] = np.log1p(df[column])

    targets = torch.tensor(df['label'].values)
    df.drop(['label'], axis=1, inplace=True)
    randomSamples = torch.tensor(df.values, requires_grad=False).float()
    
    return randomSamples, targets