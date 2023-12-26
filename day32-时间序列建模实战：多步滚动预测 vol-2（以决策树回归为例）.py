# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:59:15 2023

@author: maihuanzhuo
"""

#时间序列建模实战：多步滚动预测 vol-2（以决策树回归为例）
#删除一半的输入数据集

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模') ##修改路径

import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
data['time'] = pd.to_datetime(data['time'], format='%b-%y')

n = 6  # 使用前6个数据点
m = 2  # 预测接下来的2个数据点

# 创建滞后期特征
for i in range(n, 0, -1):
    data[f'lag_{i}'] = data['incidence'].shift(n - i + 1)

data = data.dropna().reset_index(drop=True)

train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
validation_data = data[(data['time'] >= '2012-01-01') & (data['time'] <= '2012-12-31')]

# 只对X_train、y_train、X_validation取奇数行,也就是我们说的删除一半的输入数据集。
X_train = train_data[[f'lag_{i}' for i in range(1, n+1)]].iloc[::2].reset_index(drop=True)

# 创建m个目标变量
y_train_list = [train_data['incidence'].shift(-i) for i in range(m)]
y_train = pd.concat(y_train_list, axis=1)
y_train.columns = [f'target_{i+1}' for i in range(m)]
y_train = y_train.iloc[::2].reset_index(drop=True).dropna()
X_train = X_train.head(len(y_train))

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]].iloc[::2].reset_index(drop=True)
y_validation = validation_data['incidence']

#建模与预测
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
#{'max_depth': None, 'min_samples_leaf': 3, 'min_samples_split': 8}
best_tree_model = DecisionTreeRegressor(**best_params)
best_tree_model.fit(X_train, y_train)

# 预测验证集
y_validation_pred = []

for i in range(len(X_validation)):
    pred = best_tree_model.predict([X_validation.iloc[i]])
    y_validation_pred.extend(pred[0])

y_validation_pred = np.array(y_validation_pred)[:len(y_validation)]

mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

print(mae_validation, mape_validation, mse_validation, rmse_validation)
#0.010023442460317462 0.8937068559673388 0.00017366924409377362 0.013178362724321015
# 预测训练集
y_train_pred = []

for i in range(len(X_train)):
    pred = best_tree_model.predict([X_train.iloc[i]])
    y_train_pred.extend(pred[0])

# 因为预测了多步，为了与y_train的形状相匹配，只取第一列
y_train_pred = np.array(y_train_pred)[:y_train.shape[0]]

mae_train = mean_absolute_error(y_train.iloc[:, 0], y_train_pred)
mape_train = np.mean(np.abs((y_train.iloc[:, 0] - y_train_pred) / y_train.iloc[:, 0]))
mse_train = mean_squared_error(y_train.iloc[:, 0], y_train_pred)
rmse_train = np.sqrt(mse_train)

print(mae_train, mape_train, mse_train, rmse_train)
#0.028678277777777778 1.2338229637589504 0.0016262165470818971 0.040326375327840926
