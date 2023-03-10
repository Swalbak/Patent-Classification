import pandas as pd
import numpy as np
import math

train_path = 'D:\coding\git_repositories\RoBERTa\csv_data\copy_data.csv'
train_data = pd.read_csv(train_path, index_col=False)

labels = set()
IPC_list = list(train_data["IPC Current Full (4 Characters)"])
Abstract_list = list(train_data["Abstract (Original Language)"])

# 레이블 종류 구하기
for i in range(len(IPC_list)):
    if pd.isna(IPC_list[i]):
        continue
    IPC_list[i] = list(map(lambda x: x.strip(), IPC_list[i].split('|')))
    labels.update(IPC_list[i])

labels=list(labels)

# 레이블 표시
for label in labels:
    train_data[label] = np.where(train_data['IPC Current Full (4 Characters)'].str.contains(label), 1, 0)

# 검증
print(IPC_list[0])
print(train_data.loc[0][labels].sum())