# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:31:02 2023

@author: maihuanzhuo
"""

#时间序列建模实战：Adaboost回归建模
import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模') ##修改路径

import pandas as pd
import numpy as np

# 数据读取和预处理
data = pd.read_csv('data.csv')

# 将时间列转换为日期格式
data['time'] = pd.to_datetime(data['time'], format='%b-%y')

# 拆分输入和输出
lag_period = 6

# 创建滞后期特征
for i in range(lag_period, 0, -1):
    data[f'lag_{i}'] = data['incidence'].shift(lag_period - i + 1)
    
# 删除包含NaN的行
data = data.dropna().reset_index(drop=True)

# 划分训练集和验证集
train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
validation_data = data[(data['time'] >= '2012-01-01') & (data['time'] <= '2012-12-31')]

# 定义特征和目标变量
X_train = train_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
y_train = train_data['incidence']

X_validation = validation_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
y_validation = validation_data['incidence']

# Adaboost回归建模
# sklearn.ensemble.AdaBoostRegressor(estimator=None, *, #分类器默认使用决策树分类器，而回归器默认使用决策树回归器。
#                                    n_estimators=50, #最大的弱学习器数量。最大迭代次数
#                                    learning_rate=1.0, #按指定的学习率缩小每个弱学习器的贡献。
#                                    loss='linear', #loss: 在增加新的弱学习器时用于更新权重的损失函数。可选的值包括 'linear', 'square', 和 'exponential'。
#                                    random_state=None, 
#                                    ase_estimator='deprecated')  


from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

# 初始化AdaBoostRegressor模型
ada_model = AdaBoostRegressor()

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'loss': ['linear', 'square', 'exponential']
}

# 初始化网格搜索
grid_search = GridSearchCV(ada_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
best_params
#{'learning_rate': 0.5, 'loss': 'exponential', 'n_estimators': 100}

# 使用最佳参数初始化ada模型
best_ada_model = AdaBoostRegressor(**best_params)

# 在训练集上训练模型
best_ada_model.fit(X_train, y_train)

# 对于验证集，我们需要迭代地预测每一个数据点
y_validation_pred = []

for i in range(len(X_validation)):
    if i == 0:
        pred = best_ada_model.predict([X_validation.iloc[0]])
    else:
        new_features = list(X_validation.iloc[i, 1:]) + [pred[0]]
        pred = best_ada_model.predict([new_features])
    y_validation_pred.append(pred[0])

y_validation_pred = np.array(y_validation_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 计算验证集上的MAE, MAPE, MSE和RMSE
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

# 计算训练集上的MAE, MAPE, MSE和RMSE
y_train_pred = best_ada_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

print("Train Metrics:", mae_train, mape_train, mse_train, rmse_train)
#Train Metrics: 0.0054974397979442174 0.34053655361755 4.2164213747536054e-05 0.006493397704402223
print("Validation Metrics:", mae_validation, mape_validation, mse_validation, rmse_validation)
#Validation Metrics: 0.019172146313938255 2.0176813116688774 0.0007117472722581783 0.026678592021659956

#多步滚动预测
#同样用前面6个预测后面2个值
#AdaBoostRegressor预期的目标变量y应该是一维数组，无法做多步滚动

#建立m个AdaBoostRegressor模型预测m个值

import pandas as pd
import numpy as np

# 数据读取和预处理
data = pd.read_csv('data.csv')
data_y = pd.read_csv('data.csv')
data['time'] = pd.to_datetime(data['time'], format='%b-%y')
data_y['time'] = pd.to_datetime(data_y['time'], format='%b-%y')

n = 6

#创建滞后特征
for i in range(n, 0, -1):
    data[f'lag_{i}'] = data['incidence'].shift(n - i + 1)

data = data.dropna().reset_index(drop=True)
train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
X_train = train_data[[f'lag_{i}' for i in range(1, n+1)]]

m = 3

X_train_list = []
y_train_list = []

#y_train对应每个时间点
for i in range(m):
    X_temp = X_train
    y_temp = data_y['incidence'].iloc[n + i:len(data_y) - m + 1 + i]
    X_train_list.append(X_temp)
    y_train_list.append(y_temp)

#截断x_train和y_train，保持数据同样长度
for i in range(m):
    X_train_list[i] = X_train_list[i].iloc[:-(m-1)]
    y_train_list[i] = y_train_list[i].iloc[:len(X_train_list[i])]

from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

# 初始化AdaBoostRegressor模型
ada_model = AdaBoostRegressor()

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'loss': ['linear', 'square', 'exponential']
}

best_ada_models = []

for i in range(m):
    grid_search = GridSearchCV(AdaBoostRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_list[i], y_train_list[i])
    best_params = grid_search.best_params_
    print(f"Best Parameters for Model {i + 1}: {best_params}")
    best_ada_model = AdaBoostRegressor(**grid_search.best_params_)
    best_ada_model.fit(X_train_list[i], y_train_list[i])
    best_ada_models.append(best_ada_model)
# Best Parameters for Model 1: {'learning_rate': 1, 'loss': 'linear', 'n_estimators': 150}
# Best Parameters for Model 2: {'learning_rate': 1, 'loss': 'square', 'n_estimators': 50}
# Best Parameters for Model 3: {'learning_rate': 0.1, 'loss': 'square', 'n_estimators': 100}

validation_start_time = train_data['time'].iloc[-1] + pd.DateOffset(months=1)
validation_data = data[data['time'] >= validation_start_time]

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]]
y_validation_pred_list = [model.predict(X_validation) for model in best_ada_models]
y_train_pred_list = [model.predict(X_train_list[i]) for i, model in enumerate(best_ada_models)]

def concatenate_predictions(pred_list):
    concatenated = []
    for j in range(len(pred_list[0])):
        for i in range(m):
            concatenated.append(pred_list[i][j])
    return concatenated

y_validation_pred = np.array(concatenate_predictions(y_validation_pred_list))[:len(validation_data['incidence'])]
y_train_pred = np.array(concatenate_predictions(y_train_pred_list))[:len(train_data['incidence']) - m + 1]

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_validation = mean_absolute_error(validation_data['incidence'], y_validation_pred)
mape_validation = np.mean(np.abs((validation_data['incidence'] - y_validation_pred) / validation_data['incidence']))
mse_validation = mean_squared_error(validation_data['incidence'], y_validation_pred)
rmse_validation = np.sqrt(mse_validation)
print("验证集：", mae_validation, mape_validation, mse_validation, rmse_validation)
# 验证集： 0.015505427745935454 0.9827675880342394 0.00039772778835730615 0.019943113807961536

mae_train = mean_absolute_error(train_data['incidence'][:-(m-1)], y_train_pred)
mape_train = np.mean(np.abs((train_data['incidence'][:-(m-1)] - y_train_pred) / train_data['incidence'][:-(m-1)]))
mse_train = mean_squared_error(train_data['incidence'][:-(m-1)], y_train_pred)
rmse_train = np.sqrt(mse_train)
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
# 训练集： 0.027271789080927353 1.470076670041373 0.0013175709838305059 0.03629836062180365