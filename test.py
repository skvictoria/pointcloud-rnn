import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import numpy as np
import pandas as pd

# scailing
def minmaxScailing(datalist, xmean=0, xstd=0, ymean=0, ystd=0, vmean=0, vstd=0):

  
  if xstd==0 and ystd==0:
    datalist_4_scale = np.array(datalist)
    forminmax = datalist_4_scale.reshape(-1, 5)
    x_mean, y_mean, v_mean, _, _ = forminmax.mean(axis=0)
    x_std, y_std, v_std, _, _ = forminmax.std(axis = 0)
    mean_array = np.array([x_mean, y_mean, v_mean, 0, 0])
    std_array = np.array([x_std, y_std, v_std, 1, 1])
    for i in range(len(datalist)):
      datalist[i] = (datalist[i]-mean_array)/std_array
  else:
    mean_array = np.array([xmean, ymean, vmean, 0, 0])
    std_array = np.array([xstd, ystd, vstd, 1, 1])
    for i in range(len(datalist)):
      #print('datalist shape', datalist[i])
      datalist[i] = (datalist[i]-mean_array)/std_array
      #print('datalist shape after processing', datalist[i])
    x_mean=x_std=y_mean=y_std=v_mean=v_std = 0
  return datalist, x_mean, x_std,y_mean, y_std, v_mean, v_std

cnt = 12
span = 5

tmpdf = pd.read_csv('./3.csv', index_col=0).dropna()
tmpdf = np.array(tmpdf).reshape(-1,5).tolist()

# x train mean  16.9083764402504
# x train std 12.340926996071687
# y train mean  2.5850035262139994
# y train std 3.847667373339291
# v train mean  0.1639040048970263
# v train std 6.697937021866014

test_tmp,_,_,_,_,_,_ = minmaxScailing(tmpdf, 16.908, 12.341, 2.585,3.848, 0.164,6.698)

#test_tmp = test_tmp.to_numpy()
test_final = np.empty([0, cnt, 5])
    
for i in range(len(test_tmp)-cnt-span+2):
  
  test_temp = np.empty([0,5])
  for j in range(i, i+cnt):
    test_temp = np.append(test_temp, [test_tmp[j]], axis = 0)
  test_final = np.append(test_final, [test_temp], axis=0)


with tf.Session() as sess:
  saver = tf.train.import_meta_graph('/content/gdrive/My Drive/model/1596994141/lr0.001_batch128_cnt12_span5_hidden64_.meta')
  saver.restore(sess, tf.train.latest_checkpoint('/content/gdrive/My Drive/model/1596994141'))
  graph = tf.get_default_graph()
  #graph_op = graph.get_operations()
  # for op in graph_op:
  #   print(op)
  for i in range(len(test_final)):
    start = time.time()
    x = graph.get_tensor_by_name('x_:0')
  
    feed_dict ={x:test_final[i].reshape(-1, 12,5)}

  # #Now, access the op that you want to run. 
    op_to_restore = graph.get_tensor_by_name('pred:0')

    print (np.argmax(sess.run(op_to_restore,feed_dict),axis=1))
    print(time.time()-start)
  #print(np.argmax(np.array(one_hot_predictions), axis=1))
