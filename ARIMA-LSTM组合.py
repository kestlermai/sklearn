# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:43:07 2023

@author: lenovo
"""
import os
os.chdir('C:/Users/maihuanzhuo/Desktop') ##修改路径

import time
time_start = time.time()

import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import concatenate
from pandas import concat, DataFrame

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dropout, Dense

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

from statsmodels.graphics.api import qqplot 
import statsmodels.tsa.stattools as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

import matplotlib
import warnings
import statsmodels
from scipy import  stats
import tensorflow as tf

# 调用GPU加速
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
    # 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

#读入数据
#data_raw = pd.read_csv('焦作.csv', usecols=[0, 1], squeeze=True)
data_raw = pd.read_csv('焦作.csv')
features=['AQI']
data_raw=data_raw[features]

# 显示原数据
plt.figure(figsize=(10, 3))
plt.title('数据AQI')
plt.xlabel('time')
plt.ylabel('AQI')
plt.plot(data_raw, 'blue', label='AQI')
plt.legend()
plt.show()

p_min = 0
p_max = 5
d_min = 0
d_max = 2
q_min = 0
q_max = 5

results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
 
    try:
        model = sm.tsa.ARIMA(ts_train, order=(p, d, q), )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)

fig, ax = plt.subplots(figsize=(6, 4))
ax = sns.heatmap(results_bic,mask=results_bic.isnull(),ax=ax,annot=True,fmt='.2f', )
ax.set_title('BIC')
plt.show()

train_results = sm.tsa.arma_order_select_ic(data_raw, ic=['aic', 'bic'], trend='c', max_ar=p_max, max_ma=q_max)

print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)

model = sm.tsa.ARIMA(data_raw, order=(3, 2, 1))
fit = model.fit()

# 获取残差
resid = fit.resid
# 画qq图
qqplot(resid, line='q', fit=True)
plt.show()

# 获得ARIMA的预测值
preds = fit.predict(1,len(data_raw), typ='levels')
preds_pd = preds.to_frame()
preds_pd.index -= 1

arima_result = pd.DataFrame(columns=['AQI'])
arima_result['AQI'] = data_raw
arima_result['predicted'] = preds_pd
arima_result['residuals'] = arima_result['AQI'] - arima_result['predicted']

new_data = arima_result
lstm_data = new_data['residuals'][:].values.astype(float)

def series_to_supervised(data, n_in, n_out, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    agg = concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg

#生成LSTM所需要的3维数据格式,(样本数，时间步长，特征数)
def dataprepare(values,timestep):
    reframed = series_to_supervised(values,timestep, 1)#X,y

    values = reframed.values
    #划分训练集和测试集
    train = values[1:train_len, :]
    test = values[train_len:, :]
    #得到对应的X和label（即y）
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # 把输入重塑成3D格式 [样例，时间步， 特征]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    print("train_X.shape:%s train_y.shape:%s test_X.shape:%s test_y.shape:%s" % (
    train_X.shape, train_y.shape, test_X.shape, test_y.shape))
    return train_X,train_y,test_X,test_y

# 归一化处理
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler_data = scaler.fit_transform(lstm_data.reshape(-1, 1))

#划分训练集和测试集长度
train_len = int(len(data_raw) * 0.80)
test_len=len(data_raw)-train_len
print(train_len)

timestep = 13  #滑动窗口
x_train, y_train, x_test, y_test = dataprepare(scaler_data,timestep)
#打印数据形式
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

model = Sequential()
model.add(LSTM(units=50, input_shape=(x_train.shape[1], x_train.shape[2]),activation='tanh',return_sequences=True))
model.add(LSTM(units=100, input_shape=(x_train.shape[1], x_train.shape[2]),activation='tanh',return_sequences=True))
model.add(LSTM(units=100,activation='tanh',return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',  metrics=['acc'])
model.summary()
len(model.layers)

history = model.fit(x_train, y_train, epochs=100, batch_size=64, callbacks=None, validation_split=None,validation_data=None, shuffle=False, verbose=2)

plt.figure(figsize=(9, 2))
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

score2 = model.evaluate(x_train, y_train)
print(score2)
y_test_pred = model.predict(x_test)

#对x_test进行反归一化
test_X = x_test.reshape((x_test.shape[0], x_test.shape[2]))
y_test_pred = concatenate((y_test_pred, test_X[:, 1:]), axis=1)
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test_pred = y_test_pred[:, 0]

y_testy = y_test.reshape((len(y_test), 1))
y_testy = concatenate((y_testy, test_X[:, 1:]), axis=1)
y_testy = scaler.inverse_transform(y_testy)
y_testy = y_testy[:, 0]

testScore = r2_score(y_testy, y_test_pred)
print('Train Sccore %.4f R2' %(testScore))

train_data, test_data = data_raw[0:int(len(data_raw)*0.8)+timestep], data_raw[int(len(data_raw)*0.8)+timestep:]
draw_test = new_data['predicted'][-len(y_test_pred):].values.astype(float)+y_test_pred
draw_test=draw_test.reshape(-1,1)

testScore = math.sqrt(mean_squared_error(test_data, draw_test))
print('Train Sccore %.4f RMSE' %(testScore))
testScore = mean_absolute_error(test_data, draw_test)
print('Train Sccore %.4f MAE' %(testScore))
testScore = r2_score(test_data, draw_test)
print('Train Sccore %.4f R2' %(testScore))

#显示预测结果
#%matplotlib notebook
fig1 = plt.figure(figsize=(10, 4),dpi=200)
plt.plot(new_data, label="Reference", color='green')
plt.plot(range(len(x_train)+timestep+1,len(new_data)),draw_test, color='blue',label='test_Prediction')
plt.title('Prediction', size=12)
plt.legend()
plt.show()

#%matplotlib notebook
plt.figure(figsize=(10, 4),dpi=200)
plt.plot(test_data, label="Actual", color='red',linewidth=4)
plt.plot(range(len(x_train)+timestep+1,len(new_data)),draw_test, color='blue',label='Prediction',linewidth=2.5,linestyle="--")
plt.title('ARIMA-LSTM Prediction', size=15)
plt.ylabel('AQI',size=15)
plt.xlabel('time/day',size=15)
plt.legend()
plt.show()

time_end = time.time()  
time_sum = time_end - time_start  
print(time_sum)

