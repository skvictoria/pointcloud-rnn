import pandas as pd
import numpy as np
from dataAugmentation import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cnt = 10  #
span = 8  #

n_hidden = 32  # Hidden layer num of features
n_classes = 3
learning_rate = 0.01  #
lambda_loss_amount = 0.0015  #

batch_size = 8
display_iter = 1000  # show test set accuracy during training

# 데이터 받아서 csv에 정답 넣어줌
def dataAssembly(tracknum,path='',  lanechng=(0, 0, 0), turn=(0, 0, 0)):
    tmpdf = pd.read_csv(path+'{}.csv'.format(tracknum), index_col=0)
    tmpdf["ans"] = 0
    if (lanechng[0] == 1):
        tmpdf.at[lanechng[1]:lanechng[2], 'ans'] = 1
    elif (turn[0] == 1):
        tmpdf.at[turn[1]:turn[2], 'ans'] = 2
    tmpdf = tmpdf.dropna(axis=0)
    return tmpdf

# train, test data 나눠줌
def split(data, y_data):
    n_train = int(len(data) * 0.8)
    n_test = int(len(data) - n_train)

    X_test = np.array(data[n_train:])
    y_test = np.array(y_data[n_train:])
    X_train = np.array(data[:n_train])
    y_train = np.array(y_data[:n_train])

    print('train 개수 :' , n_train)
    print('test 개수 :', n_test)

    return X_train, y_train, X_test, y_test

# dataframe을 받아서 0,1,2 폴더로 나눠줌
def dataProcess_main(tmpdf, cnt, span):
    normal_data = []
    lanechng_data = []
    turn_data = []
    for i in range(0, len(tmpdf) - cnt - span):
        tmplist = []
        for j in range(i, cnt + i):
            tmplist.append(j)
        if (tmpdf.iloc[i + cnt + span, 3] == 0):
            normal_data.append(tmpdf.iloc[tmplist, [0, 1, 2]].to_numpy())
        elif (tmpdf.iloc[i + cnt + span, 3] == 1):
            lanechng_data.append(tmpdf.iloc[tmplist, [0, 1, 2]].to_numpy())
        elif (tmpdf.iloc[i + cnt + span, 3] == 2):
            turn_data.append(tmpdf.iloc[tmplist, [0, 1, 2]].to_numpy())
        else:
            continue

    for i in range(len(lanechng_data)):
        lanechng_data.append(DA_Jitter(lanechng_data[i]))
        lanechng_data.append(DA_TimeWarp(lanechng_data[i]))
        lanechng_data.append(DA_Scaling(lanechng_data[i]))

    for i in range(len(lanechng_data)):
        turn_data.append(DA_Jitter(turn_data[i]))
        turn_data.append(DA_TimeWarp(turn_data[i]))
        turn_data.append(DA_Scaling(turn_data[i]))

    y_data = np.full((1,len(normal_data)),0)
    y_data = np.append(y_data, np.full((1, len(lanechng_data)), 1))
    y_data = np.append(y_data, np.full((1, len(turn_data)), 2))

    X_train, X_test, y_train, y_test = train_test_split(normal_data+lanechng_data+turn_data, y_data, test_size=0.2, shuffle=True, stratify=grp, random_state = 1004)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(X_train)
    print("Min:", np.min(x_train_scaled))
    print("Max:", np.max(x_train_scaled))
    x_test_scaled = x_scaler.transform(X_test)

    return x_train_scaled, y_train, x_test_scaled, y_test
