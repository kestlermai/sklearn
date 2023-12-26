# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:19:29 2023

@author: maihuanzhuo
"""

#时间序列建模实战：xgboost回归建模
#print(SVR.__doc__)查看函数文档
from xgboost import XGBRegressor
#print(XGBRegressor.__doc__)
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
Xgboost回归
回归：目标通常是预测一个连续的输出值，因此默认的目标函数是均方误差。
objective: reg:squarederror

回归：常用的评估指标如下
rmse: 均方根误差
mae: 平均绝对误差

分类：目标是预测类别。对于二分类问题，使用逻辑回归；对于多分类问题，使用多项式逻辑回归。
二分类：objective: binary:logistic
多分类：objective: multi:softprob or multi:softmax（需设置 num_class）（b）评估指标：

分类：常用的评估指标如下
error: 分类误差
logloss: 对数损失（用于二分类）
mlogloss: 多类别的对数损失（用于多分类）
auc: ROC曲线下的面积

常用参数：
max_depth: 决策树的最大深度。
learning_rate: 学习率或步长。
subsample: 训练每棵树时使用的样本的比例。
colsample_bytree: 构建每棵树时使用的特征的比例。
n_estimators: 提升迭代的次数或树的数量。
'''
from xgboost import XGBRegressor
#print(XGBRegressor.__doc__)
from sklearn.model_selection import GridSearchCV

# 初始化 XGBRegressor 模型
xgboost_model = XGBRegressor()

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'max_depth': [4, 6, 8],
    'objective': ['reg:squarederror']
}

# 初始化网格搜索
grid_search = GridSearchCV(xgboost_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
#{'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 150, 'objective': 'reg:squarederror'}

# 使用最佳参数初始化 XGBRegressor 模型
best_xgboost_model = XGBRegressor(**best_params)

# 在训练集上训练模型
best_xgboost_model.fit(X_train, y_train)

# 对于验证集，我们需要迭代地预测每一个数据点
y_validation_pred = []

for i in range(len(X_validation)):
    if i == 0:
        pred = best_xgboost_model.predict(np.array([X_validation.iloc[0]]))
    else:
        new_features = np.array([list(X_validation.iloc[i, 1:]) + [pred[0]]])
        pred = best_xgboost_model.predict(new_features)
    y_validation_pred.append(pred[0])

y_validation_pred = np.array(y_validation_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 计算验证集上的MAE, MAPE, MSE 和 RMSE
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

# 计算训练集上的MAE, MAPE, MSE 和 RMSE
y_train_pred = best_xgboost_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

print("Train Metrics:", mae_train, mape_train, mse_train, rmse_train)
#Train Metrics: 0.0017729218550357555 0.12297621444550552 5.567304183634193e-06 0.0023595135480929522
print("Validation Metrics:", mae_validation, mape_validation, mse_validation, rmse_validation)
#Validation Metrics: 0.020693434797748923 1.988412662010263 0.0009626428453114966 0.0310264861902133

#多步滚动预测
#对于Xgboost回归，目标变量y_train不能是多列的DataFrame

#建立m个XGBRegressor回归模型预测m个值

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

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'max_depth': [4, 6, 8],
    'objective': ['reg:squarederror']
}

best_xgboost_models = []

for i in range(m):
    grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')  # 使用XGBRegressor
    grid_search.fit(X_train_list[i], y_train_list[i])
    best_xgboost_model = XGBRegressor(**grid_search.best_params_)
    best_xgboost_model.fit(X_train_list[i], y_train_list[i])
    best_xgboost_models.append(best_xgboost_model)

validation_start_time = train_data['time'].iloc[-1] + pd.DateOffset(months=1)
validation_data = data[data['time'] >= validation_start_time]

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]]
y_validation_pred_list = [model.predict(X_validation) for model in best_xgboost_models]
y_train_pred_list = [model.predict(X_train_list[i]) for i, model in enumerate(best_xgboost_models)]

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
#验证集： 0.015303439232483508 1.1467118893897468 0.0003730367965542991 0.019314160519015553

mae_train = mean_absolute_error(train_data['incidence'][:-(m-1)], y_train_pred)
mape_train = np.mean(np.abs((train_data['incidence'][:-(m-1)] - y_train_pred) / train_data['incidence'][:-(m-1)]))
mse_train = mean_squared_error(train_data['incidence'][:-(m-1)], y_train_pred)
rmse_train = np.sqrt(mse_train)
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.027985439266765662 1.557820191529318 0.0014095753020031186 0.03754431118029892
