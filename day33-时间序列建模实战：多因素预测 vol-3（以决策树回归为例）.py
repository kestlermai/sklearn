# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:01:23 2023

@author: maihuanzhuo
"""

#时间序列建模实战：多因素预测 vol-3（以决策树回归为例）
#第三期为多模型预测
#我们还是使用3个数值去预测2个数值。不同的是，这2个数值分别是由2个不同参数的模型（这里都是决策树）进行预测的。
#第一个模型专门被训练来预测偶数位的数值。
#第二个模型专门被训练来预测奇数位的数值。
#假设使用前n个数值去预测下m个数值。如果m=3时，那么就需要构建3个模型，分别预测3个数值，然后依次把这3个数值按顺序拼接在一起。
#如果m=4时，那么就需要构建4个模型，分别预测4个数值，然后依次把这4个数值按顺序拼接在一起。
#同理，如果m=d时，那么就需要构建d个模型，分别预测d个数值，然后依次把这d个数值按顺序拼接在一起。以此类推。

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模') ##修改路径

import pandas as pd
import numpy as np

# 数据读取和预处理
data = pd.read_csv('data.csv')
data_y = pd.read_csv('data.csv')
data['time'] = pd.to_datetime(data['time'], format='%b-%y')
data_y['time'] = pd.to_datetime(data_y['time'], format='%b-%y')

n = 6

for i in range(n, 0, -1):
    data[f'lag_{i}'] = data['incidence'].shift(n - i + 1)

data = data.dropna().reset_index(drop=True)
train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
X_train = train_data[[f'lag_{i}' for i in range(1, n+1)]]

m = 3

X_train_list = []
y_train_list = []

#循环多个模型的特征数据
for i in range(m):
    X_temp = X_train#将X_train赋值给X_temp。这意味着每个模型的特征数据都相同。
    #这是获取标签数据的关键步骤。它使用iloc来获取一个子集，这个子集的起始点根据循环的迭代而变化。起始点是n + i，而终止点是len(data_y) - m + 1 + i。
    y_temp = data_y['incidence'].iloc[n + i:len(data_y) - m + 1 + i]
    X_train_list.append(X_temp)#将X_temp添加到X_train_list。
    y_train_list.append(y_temp)#将y_temp添加到y_train_list。
    
#对于第一个模型（i=0），我们从第n个数据点开始选择标签。
#对于第二个模型（i=1），我们从第n+1个数据点开始选择标签。
#对于第三个模型（i=2），我们从第n+2个数据点开始选择标签。

# 截断y_train使其与X_train的长度匹配
for i in range(m):
    X_train_list[i] = X_train_list[i].iloc[:-(m-1)]#这行代码将X_train_list中的每个元素（即特征数据）从末尾截断m-1行。例如，如果m=3，则截断最后2行。
    y_train_list[i] = y_train_list[i].iloc[:len(X_train_list[i])]#这行代码确保标签数据的长度与特征数据的长度相匹配。
#综上所述，我们得到的X_train_list包含三个相同的输入集（A\B\C）；同样，y_train_list包含三个输出集（D\E\F），注意D\E\F的数据不一样。
#A和D用于训练模型一，B和E用于训练模型二，C和F用于训练模型三。

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
# 模型训练
tree_model = DecisionTreeRegressor()
param_grid = {
    'max_depth': [None, 3, 5, 7, 9],
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 11)
}

best_tree_models = []

for i in range(m):
    grid_search = GridSearchCV(tree_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_list[i], y_train_list[i])
    best_tree_model = DecisionTreeRegressor(**grid_search.best_params_)
    best_tree_model.fit(X_train_list[i], y_train_list[i])
    best_tree_models.append(best_tree_model)

best_params = grid_search.best_params_
best_params
#{'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 10}

# 为了使validation_data的划分遵循上述的逻辑，我们首先需要确定其开始的时间点
# 这是在train_data最后一个时间点之后的第一个时间点
validation_start_time = train_data['time'].iloc[-1] + pd.DateOffset(months=1)
validation_data = data[data['time'] >= validation_start_time]

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]]
y_validation_pred_list = [model.predict(X_validation) for model in best_tree_models]
y_train_pred_list = [model.predict(X_train_list[i]) for i, model in enumerate(best_tree_models)]

#结果拼接：
def concatenate_predictions(pred_list):
    concatenated = []
    for j in range(len(pred_list[0])):
        for i in range(m):
            concatenated.append(pred_list[i][j])
    return concatenated
#concatenate_predictions 函数将多个模型的预测结果按照一定的顺序串联起来。
#假设我们有3个模型（即m=3），每个模型都为3个月份进行预测。那么，模型的预测列表 pred_list 可能如下所示：
#pred_list = [
#    [0.1, 0.2, 0.3],  # 模型1的预测结果
#    [0.4, 0.5, 0.6],  # 模型2的预测结果
#    [0.7, 0.8, 0.9]   # 模型3的预测结果
#]
#模型1的第1个月预测，模型2的第1个月预测，模型3的第1个月预测，模型1的第2个月预测，模型2的第2个月预测，模型3的第2个月预测，依此类推。
#[0.1, 0.4, 0.7, 0.2, 0.5, 0.8, 0.3, 0.6, 0.9]

y_validation_pred = np.array(concatenate_predictions(y_validation_pred_list))[:len(validation_data['incidence'])]
y_train_pred = np.array(concatenate_predictions(y_train_pred_list))[:len(train_data['incidence']) - m + 1]


mae_validation = mean_absolute_error(validation_data['incidence'], y_validation_pred)
mape_validation = np.mean(np.abs((validation_data['incidence'] - y_validation_pred) / validation_data['incidence']))
mse_validation = mean_squared_error(validation_data['incidence'], y_validation_pred)
rmse_validation = np.sqrt(mse_validation)
print("验证集：", mae_validation, mape_validation, mse_validation, rmse_validation)
#验证集： 0.015520779124579125 1.0369949936614837 0.00038480589242033235 0.01961646992759738

mae_train = mean_absolute_error(train_data['incidence'][:-(m-1)], y_train_pred)
mape_train = np.mean(np.abs((train_data['incidence'][:-(m-1)] - y_train_pred) / train_data['incidence'][:-(m-1)]))
mse_train = mean_squared_error(train_data['incidence'][:-(m-1)], y_train_pred)
rmse_train = np.sqrt(mse_train)
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.025700846042404565 1.3374907287799753 0.0011545433088743055 0.03397857131891077

