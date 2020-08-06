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
def dataAssembly(tracknum, lanechng, turn, path=''):
    tmpdf = pd.read_csv(path+'{}.csv'.format(tracknum), index_col=0)
    tmpdf["ans"] = 0
    ans1 = ans2 = 0
    if (lanechng[0] == 1):
        tmpdf.at[lanechng[1]:lanechng[2], 'ans'] = 1
        print('1 updated')
    if (turn[0] == 1):
        tmpdf.at[turn[1]:turn[2], 'ans'] = 2
        print('2 updated')
    tmpdf = tmpdf.dropna(axis=0)
    return tmpdf
'''
# train, test data 나눠줌
def split(data, y_data):
    n_train = int(len(data) * 0.8)
    n_test = int(len(data) - n_train)

    X_test = np.array(data[n_train:])
    y_test = np.array(y_data[n_train:])
    X_train = np.array(data[:n_train])
    y_train = np.array(y_data[:n_train])

    print('train 개수 :', n_train)
    print('test 개수 :', n_test)

    return X_train, y_train, X_test, y_test
'''

# scailing
def minmaxScailing(dataframe, xmean=0, xstd=0, ymean=0, ystd=0, vmean=0, vstd=0):
  if xstd==0 and ystd==0:
    x = dataframe[0]
    print(x)
    x_mean = x.mean()
    print(x_mean)
    x_std = x.std()
    print(x_std)
    x = (x-x_mean)/x_std
    print(x)
    y = dataframe[1]
    y_mean = y.mean()
    y_std = y.std()
    y = (y-y_mean)/y_std
    v = dataframe[2]
    v_mean = v.mean()
    v_std = v.std()
    v = (v-v_mean)/v_std
  else:
    x = dataframe[0]
    x = (x-xmean)/xstd
    y = dataframe[1]
    y = (y-ymean)/ystd
    v = dataframe[2]
    v = (v-vmean)/vstd
    x_mean=x_std=y_mean=y_std=v_mean=v_std = 0
  return pd.concat([x,y,v,dataframe[3],dataframe[4]], axis=1), x_mean, x_std,y_mean, y_std, v_mean, v_std
  

# dataframe을 받아서 0,1,2 폴더로 나눠줌
def dataProcess_main(tmpdf, cnt, span):
    normal_data = []
    lanechng_data = []
    turn_data = []
    for i in range(0, len(tmpdf) - cnt - span):
        tmplist = []
        for j in range(i, cnt + i):
            tmplist.append(j)
        if (tmpdf.iloc[i + cnt + span, 5] == 0):
            normal_data.append(tmpdf.iloc[tmplist, [0, 1, 2, 3, 4]].to_numpy())
        elif (tmpdf.iloc[i + cnt + span, 5] == 1):
            lanechng_data.append(tmpdf.iloc[tmplist, [0, 1, 2, 3, 4]].to_numpy())
        elif (tmpdf.iloc[i + cnt + span, 5] == 2):
            turn_data.append(tmpdf.iloc[tmplist, [0, 1, 2, 3, 4]].to_numpy())
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

    X_train, X_test, y_train, y_test = train_test_split(normal_data+lanechng_data+turn_data, y_data, test_size=0.2,
                                                        shuffle=True, stratify=y_data, random_state = 1004)
    print(X_train[0].shape)
    print(type(X_train[0]))
    print(X_test[0].shape)
    print(y_train[0])
    print(y_test[0])

    
    forminmax = X_train[0].tolist()
    for i in range(1, len(X_train)):
      forminmax += X_train[i].tolist()

    fortest = X_test[0].tolist()
    for i in range(1, len(X_test)):
      fortest += X_test[i].tolist()


    x_train_raw, xmean, xstd, ymean, ystd, vmean, vstd = minmaxScailing(pd.DataFrame(forminmax))
    x_train_raw = x_train_raw.to_numpy()
    x_train_scaled = np.empty([0, cnt, 5])
    
    for i in range(int(len(x_train_raw)/cnt)):
      i = i*cnt
      x_tmp = np.empty([0,5])
      for j in range(i, i+cnt):
        x_tmp = np.append(x_tmp, [x_train_raw[j]], axis = 0)
      x_train_scaled = np.append(x_train_scaled, [x_tmp], axis=0)
    x_train_scaled.tolist()

    x_test_raw, _, _, _, _, _, _ = minmaxScailing(pd.DataFrame(fortest), xmean, xstd, ymean, ystd, vmean, vstd)
    x_test_raw = x_test_raw.to_numpy()
    x_test_scaled = np.empty([0, cnt, 5])
    
    for i in range(int(len(x_test_raw)/cnt)):
      i = i*cnt
      x_tmp = np.empty([0,5])
      for j in range(i, i+cnt):
        x_tmp = np.append(x_tmp, [x_test_raw[j]], axis = 0)
      x_test_scaled = np.append(x_test_scaled, [x_tmp], axis=0)
    x_test_scaled.tolist()
    return x_train_scaled, y_train, x_test_scaled, y_test
