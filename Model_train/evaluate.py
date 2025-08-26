from math import sqrt
import pandas as pd 
import tensorflow as tf
import gc 
import gzip
import matplotlib.pyplot as plt
import numpy as np
# import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
# import xarray as xr
import os 
import math
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import r2_score,accuracy_score,precision_score
from sklearn.metrics import mean_squared_error
# 评估函数定义
# 异常相关系数
def acc(actual, predicted):
    pred_avg = np.average(predicted)

    act_avg = np.average(actual)
    diff_pred = predicted - pred_avg
    diff_act = actual - act_avg
    numerator = np.mean(np.sum(diff_pred*diff_act, axis=0))
    denominator = math.sqrt(np.mean(np.sum(diff_pred**2, axis=0)) * np.mean(np.sum(diff_act**2, axis=0)))
    ret_val = numerator/denominator
    return (100 * ret_val)


# 查看损失
def look_loss(history):
    plt.figure(figsize=(6.3, 2.5), dpi = 100)
    plt.plot(history.history['loss'], label='training data')
    plt.plot(history.history['val_loss'], label='validation data')
    plt.title('Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Number of epochs')
    plt.grid()
    plt.legend(loc="upper right")
    plt.show()


# 模型预测
def all_estimate(best_model,x_test,y_test):
    testPred_1 = best_model.predict(x_test)
    a = y_test
    y_test_p = a.reshape(-1,1)
    testPred_p = testPred_1.reshape(-1,1)

    # 改
    y_test_p = y_test_p[~np.isnan(y_test_p)]
    testPred_p = testPred_p[~np.isnan(testPred_p)]


    rmse = sqrt(mean_squared_error(y_test_p,testPred_p))
    print('Test RMSE: %.3f' % rmse)

    print('acc:',acc(y_test_p,testPred_p))

    r2 = r2_score(y_test_p,testPred_p)
    print("R² score:", r2)  
    

def respective_estimate(best_model,x_test,y_test):
    re2_list = []
    rmse_list = []
    for i in range(6):
        # 模型预测
        testPred = best_model.predict(x_test[i:i+1])
        a = y_test[i:i+1]
        y_test_p = a.reshape(-1,1)
        testPred_p = testPred.reshape(-1,1)
        r2 = r2_score(y_test_p,testPred_p)
        rmse = sqrt(mean_squared_error(y_test_p,testPred_p))
        re2_list.append(r2)
        rmse_list.append(rmse)    
    return re2_list,rmse_list