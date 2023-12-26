# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:07:43 2023

@author: maihuanzhuo
"""

#时间序列建模实战：LightGMB回归建模
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

#LightGBM
#核心参数：如 boosting_type、num_boost_round、learning_rate 等。
#boosting_type:['gbdt', 'dart', 'goss'],不同类型的渐变增强提升方法，gbdt梯度提升决策树，dart梯度提升，lgbm goss基于梯度的单边采样。
# 标准的gbdt是可靠的，但在大型数据集上速度不够快。因此goss提出了一种基于梯度的采样方法来避免搜索整个搜索空间。
# 我们知道，对于每个数据实例，当梯度很小时，这意味着不用担心数据是经过良好训练的，而当梯度很大时，应该重新训练。
# 这里我们有两个方面，数据实例有大的和小的渐变。因此，goss以一个大的梯度保存所有数据，
# 并对一个小梯度的数据进行随机抽样(这就是为什么它被称为单边抽样)。这使得搜索空间更小，goss的收敛速度更快。
#num_boost_round的别名num_trees，即同一参数。

#LightGBM的调参策略
# learning_rate：一般先设定为0.1，最后再作调整，合适候选值为：[0.01, 0.015, 0.025, 0.05, 0.1]；
# max_depth：树的最大深度，默认值为-1，表示不做限制，合理的设置可以防止过拟合；
# num_leaves：叶子的个数，默认值为31，此参数的数值应该小于2^max_depth；
# min_data_in_leaf /min_child_samples：设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合，默认值20；
# min_split_gain：默认值为0，设置的值越大，模型就越保守，推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]；
# subsample：选择小于1的比例可以防止过拟合，但会增加样本拟合的偏差，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]；
# colsample_bytree：特征随机采样的比例，默认值为1，推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]；
# reg_alpha：推荐的候选值为：[0, 0.01~0.1, 1]；
# reg_lambda：推荐的候选值为：[0, 0.1, 0.5, 1]；
# 参考大佬的调参策略：
# learning_rate设置为0.1；
# 调参：max_depth, num_leaves, min_data_in_leaf, min_split_gain, subsample, colsample_bytree；
# 调参：reg_lambda , reg_alpha；
# 降低学习率，继续调整参数，学习率合适候选值为：[0.01, 0.015, 0.025, 0.05, 0.1]；

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
# 初始化 LGBMRegressor 模型
lgbm_model = LGBMRegressor()

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'num_leaves': [31, 50, 75],
    'boosting_type': ['gbdt', 'dart', 'goss']
}

# 初始化网格搜索
grid_search = GridSearchCV(lgbm_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
best_params
#{'boosting_type': 'dart', 'learning_rate': 0.5, 'n_estimators': 50, 'num_leaves': 31}

# 使用最佳参数初始化 LGBMRegressor 模型
best_lgbm_model = LGBMRegressor(**best_params)

# 在训练集上训练模型
best_lgbm_model.fit(X_train, y_train)

# 对于验证集，我们需要迭代地预测每一个数据点(单步滚动预测)
y_validation_pred = []

for i in range(len(X_validation)):
    if i == 0:
        pred = best_lgbm_model.predict([X_validation.iloc[0]])
    else:
        new_features = list(X_validation.iloc[i, 1:]) + [pred[0]]
        pred = best_lgbm_model.predict([new_features])
    y_validation_pred.append(pred[0])

y_validation_pred = np.array(y_validation_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 计算验证集上的MAE, MAPE, MSE 和 RMSE
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

# 计算训练集上的MAE, MAPE, MSE 和 RMSE
y_train_pred = best_lgbm_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

print("Train Metrics:", mae_train, mape_train, mse_train, rmse_train)
#Train Metrics: 0.006820985456644797 0.3553927260149519 7.828676683331904e-05 0.008847980946708635
print("Validation Metrics:", mae_validation, mape_validation, mse_validation, rmse_validation)
#Validation Metrics: 0.022923016875929948 2.661594945770784 0.001025165459650328 0.03201820512849413

#多步滚动预测 
# 对于LGBMRegressor，目标变量y_train不能是多列的DataFrame
import pandas as pd
import numpy as np

# 数据读取和预处理
data = pd.read_csv('data.csv')

# 将时间列转换为日期格式
data['time'] = pd.to_datetime(data['time'], format='%b-%y')

n = 6  
m = 2 

# 创建滞后期特征
for i in range(n, 0, -1):
    data[f'lag_{i}'] = data['incidence'].shift(n - i + 1)

data = data.dropna().reset_index(drop=True)

train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
validation_data = data[(data['time'] >= '2012-01-01') & (data['time'] <= '2012-12-31')]

X_train = train_data[[f'lag_{i}' for i in range(1, n+1)]]

# 创建m个目标变量
y_train_list = [train_data['incidence'].shift(-i) for i in range(m)]#创建m列incidence训练集，第二列肯定是比前一列少一位元素
y_train = pd.concat(y_train_list, axis=1)#合并两列
y_train.columns = [f'target_{i+1}' for i in range(m)]#对m列进行分别命名
y_train = y_train.dropna()#因为m列肯定比前一列少一位元素，那么需要对齐数据，去除缺失值

X_train = X_train.iloc[:-m+1, :]#同样的，X训练集也要对齐，需要去除最后m-1行的数据

#创建验证集
X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]]
y_validation = validation_data['incidence']

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
# 初始化 LGBMRegressor 模型
lgbm_model = LGBMRegressor()

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'num_leaves': [31, 50, 75],
    'boosting_type': ['gbdt', 'dart', 'goss']
}

grid_search = GridSearchCV(lgbm_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_params
#{'max_depth': 5, 'n_estimators': 50}
best_rf_model = LGBMRegressor(**best_params)
best_rf_model.fit(X_train, y_train)
#ValueError: DataFrame for label cannot have multiple columns
# 对于LGBMRegressor，目标变量y_train不能是多列的DataFrame，不能做多步滚动预测


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

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

# 初始化AdaBoostRegressor模型
lgbm_model = LGBMRegressor()

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'num_leaves': [31, 50, 75],
    'boosting_type': ['gbdt', 'dart', 'goss']
}


best_lgbm_models = []

for i in range(m):
    grid_search = GridSearchCV(LGBMRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_list[i], y_train_list[i])
    best_params = grid_search.best_params_
    best_lgbm_model = LGBMRegressor(**grid_search.best_params_)
    best_lgbm_model.fit(X_train_list[i], y_train_list[i])
    best_lgbm_models.append(best_lgbm_model)


validation_start_time = train_data['time'].iloc[-1] + pd.DateOffset(months=1)
validation_data = data[data['time'] >= validation_start_time]

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]]
y_validation_pred_list = [model.predict(X_validation) for model in best_lgbm_models]
y_train_pred_list = [model.predict(X_train_list[i]) for i, model in enumerate(best_lgbm_models)]

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
# 验证集： 0.013725917250907322 1.114671130761191 0.0003469142164606173 0.018625633317034276

mae_train = mean_absolute_error(train_data['incidence'][:-(m-1)], y_train_pred)
mape_train = np.mean(np.abs((train_data['incidence'][:-(m-1)] - y_train_pred) / train_data['incidence'][:-(m-1)]))
mse_train = mean_squared_error(train_data['incidence'][:-(m-1)], y_train_pred)
rmse_train = np.sqrt(mse_train)
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
# 训练集： 0.024812185403776024 1.3333162834819863 0.0010509333517661388 0.03241810222338962

