# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:05:07 2023

@author: maihuanzhuo
"""
#时间序列建模实战：多因素预测 vol-2（以决策树回归为例）
#使用决策树回归的单步滚动预测

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模/时间序列建模实战：多因素预测---ARIMAX') ##修改路径

 #读取数据（风速以及其他额外变量）
import numpy as np
import pandas as pd 
data = pd.read_csv('wind_dataset.csv')

# 将日期列转换为日期格式
data['DATE'] = pd.to_datetime(data['DATE'])

#中位数填充
data = data.fillna(data.median())

# 删除不必要的列
data = data.drop(columns=['IND', 'IND.1', 'IND.2'])

# 创建WIND滞后期特征
lag_period = 6
for i in range(lag_period, 0, -1):
    data[f'WIND_lag_{i}'] = data['WIND'].shift(lag_period - i + 1)
    
#创建滞后特征
data['T.MIN_lag3'] = data['T.MIN'].shift(3)
data['T.MIN.G_lag3'] = data['T.MIN.G'].shift(3)

# 删除包含NaN的行
data = data.dropna().reset_index(drop=True)

#划分训练集和验证集
train_data = data[data['DATE'] < data['DATE'].max() - pd.DateOffset(years=1)]#选取最后一年之前的作为训练集
validation_data = data[data['DATE'] >= data['DATE'].max() - pd.DateOffset(years=1)]#选取最后一年作为验证集

# 定义特征和目标变量
features = ['RAIN', 'T.MAX', 'T.MIN_lag3', 'T.MIN.G_lag3'] + [f'WIND_lag_{i}' for i in range(1, lag_period + 1)]#定义外部变量和滞后期特征
X_train = train_data[features]
y_train = train_data['WIND']

X_validation = validation_data[features]
y_validation = validation_data['WIND']

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
# 初始化决策树模型，并使用网格搜索寻找最佳参数
tree_model = DecisionTreeRegressor()
param_grid = {
    'max_depth': [None, 3, 5, 7, 9],
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 11)
}
grid_search = GridSearchCV(tree_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_params 
#{'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 4}

# 使用最佳参数初始化决策树模型并在训练集上训练
best_tree_model = DecisionTreeRegressor(**best_params)
best_tree_model.fit(X_train, y_train)

# 使用滚动预测的方式预测验证集上的风速
y_validation_pred = []
for i in range(len(X_validation)):
    if i < 6:
        pred = best_tree_model.predict([X_validation.iloc[i]])##当i<6时，直接使用验证集的第一个样本进行预测，因为前6个时间步没有足够的历史数据来构建滞后特征
    else:
        new_features = X_validation.iloc[i][0:4].values.tolist() + y_validation_pred[-6:]#如果大于等于6，则用前面四个外部特征以及后6位WIND的滞后值特征去构建新特征
        pred = best_tree_model.predict([new_features])#然后用这些新的特征去预测下一个WIND值
    y_validation_pred.append(pred[0])#append函数在列表末尾添加下一个预测的WIND值
y_validation_pred = np.array(y_validation_pred)
#使用前6个WIND，加上1个RAIN、T.MAX、T.MIN（lag3）和T.MIN.G（lag3），一共10个特征去预测下一个WIND

from sklearn.metrics import mean_absolute_error, mean_squared_error
#计算验证集上的误差
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))

# 计算训练集上的误差
y_train_pred = best_tree_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))

mae_validation, mse_validation, rmse_validation, mape_validation
#(4.193192036437853, 26.137640345158232, 5.112498444514016, 0.8812749161618093)
mae_train, mse_train, rmse_train, mape_train
#(3.0054299616258766, 14.60846619373412, 3.8221023264342517, inf)

#RAIN、T.MAX、T.MIN（lag3）和T.MIN.G（lag3）为什么只要1个数值作训练集呢？
#而WIND考虑本身的时间序列变化的规律，因此创建了WIND的滞后期特征，WIND_lag1,..,6。
#但RAIN、T.MAX、T.MIN（lag3）和T.MIN.G（lag3）本身也具有时间序列变化规律

#读取数据（风速以及其他额外变量）
import numpy as np
import pandas as pd 
data = pd.read_csv('wind_dataset.csv')

# 将日期列转换为日期格式
data['DATE'] = pd.to_datetime(data['DATE'])

#中位数填充
data = data.fillna(data.median())

# 删除不必要的列
data = data.drop(columns=['IND', 'IND.1', 'IND.2'])

#创建滞后特征
data['T.MIN_lag3'] = data['T.MIN'].shift(3)
data['T.MIN.G_lag3'] = data['T.MIN.G'].shift(3)

data = data.dropna().reset_index(drop=True)

# 创建滞后期特征
lag_period = 6
for i in range(lag_period, 0, -1):
    data[f'WIND_lag_{i}'] = data['WIND'].shift(lag_period - i + 1)
    data[f'RAIN_lag_{i}'] = data['RAIN'].shift(lag_period - i + 1)
    data[f'T.MAX_lag_{i}'] = data['T.MAX'].shift(lag_period - i + 1)
    data[f'T.MIN_lag3_lag_{i}'] = data['T.MIN_lag3'].shift(lag_period - i + 1)
    data[f'T.MIN.G_lag3_lag_{i}'] = data['T.MIN.G_lag3'].shift(lag_period - i + 1)
#如果不同变量的滞后期不一样呢？譬如RAIN lag=3呢，如何确定每个变量最佳的lag_period呢

# 删除包含NaN的行
data = data.dropna().reset_index(drop=True)

#划分训练集和验证集
train_data = data[data['DATE'] < data['DATE'].max() - pd.DateOffset(years=1)]#选取最后一年之前的作为训练集
validation_data = data[data['DATE'] >= data['DATE'].max() - pd.DateOffset(years=1)]#选取最后一年作为验证集

# 定义特征和目标变量
features = [f'RAIN_lag_{i}' for i in range(1, lag_period + 1)] + \
    [f'T.MAX_lag_{i}' for i in range(1, lag_period + 1)]+ \
        [f'T.MIN_lag3_lag_{i}' for i in range(1, lag_period + 1)]+\
            [f'T.MIN.G_lag3_lag_{i}' for i in range(1, lag_period + 1)] + \
                [f'WIND_lag_{i}' for i in range(1, lag_period + 1)]
           
X_train = train_data[features]
y_train = train_data['WIND']

X_validation = validation_data[features]
y_validation = validation_data['WIND']

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
# 初始化决策树模型，并使用网格搜索寻找最佳参数
tree_model = DecisionTreeRegressor()
param_grid = {
    'max_depth': [None, 3, 5, 7, 9],
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 11)
}
grid_search = GridSearchCV(tree_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_params 
#{'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}

# 使用最佳参数初始化决策树模型并在训练集上训练
best_tree_model = DecisionTreeRegressor(**best_params)
best_tree_model.fit(X_train, y_train)

# 使用单步滚动预测的方式预测验证集上的风速
y_validation_pred = []
for i in range(len(X_validation)):
    if i < 6:
        pred = best_tree_model.predict([X_validation.iloc[i]])
    else:
        new_features = X_validation.iloc[i][0:24].values.tolist() + y_validation_pred[-6:]#创建一个列表外部特征变成24个和最后6个预测值（WIND）
        pred = best_tree_model.predict([new_features])
    y_validation_pred.append(pred[0])
y_validation_pred = np.array(y_validation_pred)
#这里变成一共30个特征去预测下一个WIND

from sklearn.metrics import mean_absolute_error, mean_squared_error
#计算验证集上的误差
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))

# 计算训练集上的误差
y_train_pred = best_tree_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))

mae_validation, mse_validation, rmse_validation, mape_validation
#(4.066194646474813, 25.590773579209333, 5.058732408341968, 0.8076733508003475)
mae_train, mse_train, rmse_train, mape_train
#(3.156428990023891, 15.996169552396776, 3.999521165389274, inf)

#模型性能提升了