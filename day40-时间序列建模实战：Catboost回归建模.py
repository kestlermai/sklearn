# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:52:42 2023

@author: maihuanzhuo
"""

#时间序列建模实战：Catboost回归建模

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模') ##修改路径

import pandas as pd
import numpy as np

#单步滚动预测
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

'''
Catboost回归
（a）通用参数：
learning_rate: 学习率，决定了模型每一步的步长。常用的值为0.01, 0.03, 0.1等。
iterations: 树的数量。
depth: 树的深度。
l2_leaf_reg: L2正则化项的系数。
cat_features: 分类特征的列索引列表。
loss_function: 损失函数。对于分类，常见的是Logloss（二分类）或MultiClass（多分类）。对于回归，常见的是RMSE。
border_count: 用于数值特征的分箱数量。较高的值可能会导致过拟合，较低的值可能会导致欠拟合。
verbose: 显示的训练日志的详细程度。

（b）专用于分类的参数：
classes_count: 在多分类任务中，类别的数量。
class_weights: 各类的权重，用于不平衡分类任务。
auto_class_weights: 用于处理类不平衡的自动权重计算方法。

（c）专用于回归的参数：
scale_pos_weight: 用于不平衡的回归任务。

（d）异同点：
相同点: 大部分参数（如learning_rate, depth, l2_leaf_reg等）在回归和分类任务中都是相同的，并且它们的含义和效果也是一致的。
不同点: 损失函数loss_function是根据任务（回归或分类）来确定的。
此外，某些参数（如classes_count和class_weights）仅在分类任务中有意义，而scale_pos_weight更倾向于回归任务。

此外，在使用CatBoost时，建议始终查阅其官方文档，因为该库可能会经常更新，新的参数或功能可能会被添加进来。网址如下：
https://catboost.ai/docs/
'''
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
# 初始化 CatBoostRegressor 模型
catboost_model = CatBoostRegressor(verbose=0)

# 定义参数网格
param_grid = {
    'iterations': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'depth': [4, 6, 8],
    'loss_function': ['RMSE']
}

# 初始化网格搜索
grid_search = GridSearchCV(catboost_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
best_params
#{'depth': 6, 'iterations': 150, 'learning_rate': 0.05, 'loss_function': 'RMSE'}
# 使用最佳参数初始化 CatBoostRegressor 模型
best_catboost_model = CatBoostRegressor(**best_params, verbose=0)

# 在训练集上训练模型
best_catboost_model.fit(X_train, y_train)

# 对于验证集，我们需要迭代地预测每一个数据点
y_validation_pred = []

for i in range(len(X_validation)):
    if i == 0:
        pred = best_catboost_model.predict([X_validation.iloc[0]])
    else:
        new_features = list(X_validation.iloc[i, 1:]) + [pred[0]]
        pred = best_catboost_model.predict([new_features])
    y_validation_pred.append(pred[0])

y_validation_pred = np.array(y_validation_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 计算验证集上的MAE, MAPE, MSE 和 RMSE
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

# 计算训练集上的MAE, MAPE, MSE 和 RMSE
y_train_pred = best_catboost_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

print("Train Metrics:", mae_train, mape_train, mse_train, rmse_train)
#Train Metrics: 0.0036321397637129674 0.204984328176331 2.0887324585172655e-05 0.00457026526420214
print("Validation Metrics:", mae_validation, mape_validation, mse_validation, rmse_validation)
#Validation Metrics: 0.019715298655878132 2.0563758184761163 0.0008059838072068701 0.028389853948318756

#多步滚动预测 
#对于Catboost回归，目标变量y_train不能是多列的DataFrame

#建立m个Catboost回归模型预测m个值

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

from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'iterations': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'depth': [4, 6, 8],
    'loss_function': ['RMSE']
}

best_catboost_models = []

for i in range(m):
    grid_search = GridSearchCV(CatBoostRegressor(verbose=0), param_grid, cv=5, scoring='neg_mean_squared_error')  # 使用CatBoostRegressor
    grid_search.fit(X_train_list[i], y_train_list[i])
    best_catboost_model = CatBoostRegressor(**grid_search.best_params_, verbose=0)
    best_catboost_model.fit(X_train_list[i], y_train_list[i])
    best_catboost_models.append(best_catboost_model)

validation_start_time = train_data['time'].iloc[-1] + pd.DateOffset(months=1)
validation_data = data[data['time'] >= validation_start_time]

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]]
y_validation_pred_list = [model.predict(X_validation) for model in best_catboost_models]
y_train_pred_list = [model.predict(X_train_list[i]) for i, model in enumerate(best_catboost_models)]

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
#验证集： 0.014321280317765597 1.0566209624520266 0.00033334300994586775 0.01825768358653057

mae_train = mean_absolute_error(train_data['incidence'][:-(m-1)], y_train_pred)
mape_train = np.mean(np.abs((train_data['incidence'][:-(m-1)] - y_train_pred) / train_data['incidence'][:-(m-1)]))
mse_train = mean_squared_error(train_data['incidence'][:-(m-1)], y_train_pred)
rmse_train = np.sqrt(mse_train)
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.02591816931601743 1.4243369674429265 0.0012138645275309092 0.03484055865698639