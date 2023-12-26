# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:03:05 2023

@author: maihuanzhuo
"""

#时间序列建模实战：多步滚动预测 vol-1（以决策树回归为例）
#多步滚动预测就是使用前n个数值去预测下m个数值
#假设，n = 3，m = 2，那么，之前举的例子就变成：使用三个数据点（1,2,3）来预测第4个和第5个数据点（4，5）
#使用三个数据点（2,3,4）来预测第5个和第6个数据点（5，6）
#使用三个数据点（3,4,5）来预测第6个和第7个数据点（6，7）
#这里有一个问题，预测的结果会出现重合。例如5，可以是（1,2,3）和（2,3,4）预测的结果
#（1）对于重复的预测值，取平均处理。例如，（1,2,3）预测出3.9和4.5，（2,3,4）预测出5.2和6.3，那么拼起来的结果就是3.9,（4.5 + 5.2）/2, 6.3。
#（2）删除一半的输入数据集。例如，4,5由（1,2,3）预测，6,7由（3,4,5）预测，删掉输入数据（2,3,4）。

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模') ##修改路径

import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
data['time'] = pd.to_datetime(data['time'], format='%b-%y')

n = 6  # 使用前6个数据点
m = 2  # 预测接下来的2个数据点

# 创建滞后期特征
for i in range(n, 0, -1):#从n递减到1
    data[f'lag_{i}'] = data['incidence'].shift(n - i + 1)

data = data.dropna().reset_index(drop=True)

train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
validation_data = data[(data['time'] >= '2012-01-01') & (data['time'] <= '2012-12-31')]

X_train = train_data[[f'lag_{i}' for i in range(1, n+1)]]

# 创建m个目标变量
y_train_list = [train_data['incidence'].shift(-i) for i in range(m)]
y_train = pd.concat(y_train_list, axis=1)
y_train.columns = [f'target_{i+1}' for i in range(m)]
y_train = y_train.dropna()
X_train = X_train.iloc[:-m+1, :]

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]]
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
#{'max_depth': None, 'min_samples_leaf': 10, 'min_samples_split': 2}
best_tree_model = DecisionTreeRegressor(**best_params)
best_tree_model.fit(X_train, y_train)

# 预测验证集
y_validation_pred = []

for i in range(len(X_validation) - m + 1):
    pred = best_tree_model.predict([X_validation.iloc[i]])
    y_validation_pred.extend(pred[0])
#与单步滚动预测range(len(X_validation))不同的是，多步滚动预测range(len(X_validation) - m + 1)预测值存在重叠
#在多步滚动预测中，每次预测出来都有m个值，而前m-1个预测结果都与上一次的预测结果存在重叠，因此就需要跳过m-1个时间步
# 重叠预测值取平均
for i in range(1, m):
    for j in range(len(y_validation_pred) - i):
        y_validation_pred[j+i] = (y_validation_pred[j+i] + y_validation_pred[j]) / 2
#在多次连续的预测中，除了第一次预测的结果，后面的每次预测的前m-1个数值都是与上次预测结果的最后m-1个数值重叠的
#外层循环i控制重叠的深度（从1到m-1），内层循环j则是遍历整个预测结果列表。
y_validation_pred = np.array(y_validation_pred)[:len(y_validation)]

mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

print(mae_validation, mape_validation, mse_validation, rmse_validation)
#0.0166892384631587 1.1174173733199757 0.0005950571601865544 0.02439379347675458

# 拟合训练集
y_train_pred = []

for i in range(len(X_train) - m + 1):
    pred = best_tree_model.predict([X_train.iloc[i]])
    y_train_pred.extend(pred[0])

# 重叠预测值取平均
for i in range(1, m):
    for j in range(len(y_train_pred) - i):
        y_train_pred[j+i] = (y_train_pred[j+i] + y_train_pred[j]) / 2

y_train_pred = np.array(y_train_pred)[:len(y_train)]
mae_train = mean_absolute_error(y_train.iloc[:, 0], y_train_pred)
mape_train = np.mean(np.abs((y_train.iloc[:, 0] - y_train_pred) / y_train.iloc[:, 0]))
mse_train = mean_squared_error(y_train.iloc[:, 0], y_train_pred)
rmse_train = np.sqrt(mse_train)

print(mae_train, mape_train, mse_train, rmse_train)
#0.02309069439996685 1.1010557871499014 0.0008859837024388373 0.029765478367377824
