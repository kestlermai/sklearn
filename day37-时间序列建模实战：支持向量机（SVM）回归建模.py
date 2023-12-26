# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:34:44 2023

@author: maihuanzhuo
"""

#时间序列建模实战：支持向量机（SVM）回归建模

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


#scikit-learn 提供了3种支持向量机 (SVM) 的回归器：sklearn.svm.SVR、sklearn.svm.NuSVR和sklearn.svm.LinearSVR：
#1.SVR (Support Vector Regression)
#说明：SVR是基于libsvm的支持向量回归的实现。print(SVR.__doc__)查看函数文档
# 核函数：可以使用多种核函数，例如线性、多项式、RBF（径向基函数）和sigmoid等。

# sklearn.svm.SVR(*, kernel='rbf', #使用的核函数。例如 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' 或者是一个自定义的函数。
#                 degree=3,#多项式核函数的度（只在kernel='poly'时使用）。
#                 gamma='scale',#'rbf', 'poly' 和 'sigmoid' 的核函数系数。
#                 coef0=0.0, #多项式和sigmoid核函数的独立项。
#                 tol=0.001, #控制训练过程的收敛容忍度
#                 C=1.0, #误差项的惩罚参数。C越大，模型对误差的容忍度越低。
#                 epsilon=0.1, #ε-不敏感损失函数中的ε。它指定了没有惩罚的训练样本的边界。
#                 shrinking=True, #是否使用收缩启发式。
#                 cache_size=200, #定义了内核缓存的大小
#                 verbose=False, 
#                 max_iter=-1)#指定了训练过程中的最大迭代次数


# 2.NuSVR
# 说明：NuSVR与SVR相似，但它使用了ν-SVM形式的参数化。它允许用户对支持向量的数量进行参数化。
# 核函数：与SVR相同，可以使用多种核函数。

# sklearn.svm.NuSVR(*, nu=0.5, #控制支持向量的数量。实际上是支持向量的上限和下限之间的比例。
#                   C=1.0, 
#                   kernel='rbf', 
#                   degree=3, 
#                   gamma='scale', 
#                   coef0=0.0, 
#                   shrinking=True, 
#                   tol=0.001, 
#                   cache_size=200,
#                   verbose=False, 
#                   max_iter=-1)


# 3.LinearSVR
# 说明：LinearSVR是基于liblinear的线性支持向量回归的实现。与SVR不同，它只处理线性核函数，并且通常比SVR(kernel='linear')更快。
# 核函数：仅线性。

# sklearn.svm.LinearSVR(*, epsilon=0.0, ##ε-不敏感损失函数中的ε。它指定了没有惩罚的训练样本的边界。
#                       tol=0.0001, 
#                       C=1.0, 
#                       loss='epsilon_insensitive', #指定损失函数。可选值有 'epsilon_insensitive' 和 'squared_epsilon_insensitive'。
#                       fit_intercept=True, #是否要在模型中拟合截距
#                       intercept_scaling=1.0, #用于控制截距项（intercept）的缩放，调整模型对截距项的敏感度
#                       dual='warn', #是否解决对偶问题。对于大规模数据，推荐设置为False。
#                       verbose=0, 
#                       random_state=None, 
#                       max_iter=1000)


# 异同总结：
# -SVR和NuSVR可以处理非线性问题，因为它们支持多种核函数。而LinearSVR仅处理线性问题。
# -NuSVR通过ν参数提供了对支持向量数量的控制。
# -对于具有线性核的问题，LinearSVR通常比SVR(kernel='linear')更快。
# -三者在误差控制方面都使用了epsilon参数。
# -C参数在所有三种模型中都存在，表示误差项的惩罚。
# -LinearSVR有一个独特的loss参数，而NuSVR有一个独特的nu参数。

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# 初始化SVR模型
svr_model = SVR()

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# 初始化网格搜索
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
best_params
#{'C': 0.1, 'epsilon': 0.01, 'kernel': 'rbf'}
# 使用最佳参数初始化SVR模型
best_svr_model = SVR(**best_params)

# 在训练集上训练模型
best_svr_model.fit(X_train, y_train)

# 对于验证集，我们需要迭代地预测每一个数据点
y_validation_pred = []

for i in range(len(X_validation)):
    if i == 0:
        pred = best_svr_model.predict([X_validation.iloc[0]])
    else:
        new_features = list(X_validation.iloc[i, 1:]) + [pred[0]]
        pred = best_svr_model.predict([new_features])
    y_validation_pred.append(pred[0])

y_validation_pred = np.array(y_validation_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 计算验证集上的MAE, MAPE, MSE和RMSE
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

# 计算训练集上的MAE, MAPE, MSE和RMSE
y_train_pred = best_svr_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

print("Train Metrics:", mae_train, mape_train, mse_train, rmse_train)
#Train Metrics: 0.008189903026561435 0.4513116668614909 0.0001075605089672468 0.010371138267675675
print("Validation Metrics:", mae_validation, mape_validation, mse_validation, rmse_validation)
#Validation Metrics: 0.020450583474068484 2.188477475279078 0.000916632161464467 0.03027593370095243

#SVR（支持向量机回归）与RandomForestRegressor有一些关键的不同，主要的不同之处在于SVR只对一个目标变量进行预测。
#而RandomForestRegressor可以同时对多个目标变量进行预测。
#因此无法进行多步滚动预测，SVR一次只能预测一个值

#建立m个SVR模型预测m个值

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

for i in range(m):
    X_temp = X_train
    y_temp = data_y['incidence'].iloc[n + i:len(data_y) - m + 1 + i]
    X_train_list.append(X_temp)
    y_train_list.append(y_temp)

for i in range(m):
    X_train_list[i] = X_train_list[i].iloc[:-(m-1)]
    y_train_list[i] = y_train_list[i].iloc[:len(X_train_list[i])]


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
# 模型训练
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

best_svr_models = []

for i in range(m):
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_list[i], y_train_list[i])
    best_params = grid_search.best_params_
    print(f"Best Parameters for Model {i + 1}: {best_params}")
    best_svr_model = SVR(**grid_search.best_params_)
    best_svr_model.fit(X_train_list[i], y_train_list[i])
    best_svr_models.append(best_svr_model)
# Best Parameters for Model 1: {'C': 0.1, 'epsilon': 0.01, 'kernel': 'rbf'}
# Best Parameters for Model 2: {'C': 0.1, 'epsilon': 0.01, 'kernel': 'rbf'}
# Best Parameters for Model 3: {'C': 0.1, 'epsilon': 0.01, 'kernel': 'rbf'}

validation_start_time = train_data['time'].iloc[-1] + pd.DateOffset(months=1)
validation_data = data[data['time'] >= validation_start_time]

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]]
y_validation_pred_list = [model.predict(X_validation) for model in best_svr_models]
y_train_pred_list = [model.predict(X_train_list[i]) for i, model in enumerate(best_svr_models)]

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
#验证集： 0.014412941557770684 1.1131093721572747 0.00035121580211031647 0.018740752442479903

mae_train = mean_absolute_error(train_data['incidence'][:-(m-1)], y_train_pred)
mape_train = np.mean(np.abs((train_data['incidence'][:-(m-1)] - y_train_pred) / train_data['incidence'][:-(m-1)]))
mse_train = mean_squared_error(train_data['incidence'][:-(m-1)], y_train_pred)
rmse_train = np.sqrt(mse_train)
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.023607395584098967 1.321710848805072 0.0009796698152562003 0.031299677558342356
