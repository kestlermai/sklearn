# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:40:26 2023

@author: maihuanzhuo
"""
###时间序列建模实战：单步滚动预测（以决策树回归为例）
#使用决策树建立一个时间序列预测模型
#使用2004年1月至2011年12月的数据进行训练，采用2002年1月至2012年12月的数据作为验证集。
#滞后期（lag）暂时设置为6，也就是使用前6个数字预测地7个数值。
#决策树模型采用网格搜索进行参数寻优。
import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模') ##修改路径

#读取数据
import pandas as pd
data = pd.read_csv('data.csv')
data.head()

# 将时间列转换为日期格式
data['time'] = pd.to_datetime(data['time'], format='%b-%y')
# 显示转换后的前几行数据
data.head()

#拆分输入和输出
lag_period = 6

# 创建滞后期特征
for i in range(1, lag_period + 1):#range1到7，但不包含7，所以只循环到6
    data[f'lag_{i}'] = data['incidence'].shift(i)

# 删除包含NaN的行
data = data.dropna().reset_index(drop=True)

#数据集拆分
# 划分训练集和验证集
train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
validation_data = data[(data['time'] >= '2012-01-01') & (data['time'] <= '2012-12-31')]
train_data.shape, validation_data.shape
# 定义特征和目标变量
X_train = train_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
y_train = train_data['incidence']
X_validation = validation_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
y_validation = validation_data['incidence']
X_train.shape, X_validation.shape

#但是，这其实有一个问题：验证集是不能用的。
#因为这是用前面的数据点进行预测下一个点，而下一个点则会被用来预测下下一个点，如此类推。
#假设我们有一组时间序列数据：1,2,3,4,5,6,7,8，其中滞后期设定为3
#在训练集中，我们使用前三个数据点（1,2,3）来预测第四个数据点（4），然后使用（2,3,4）来预测第五个数据点（5），直到使用（3,4,5）来预测第六个数据点（6）。
#接下来，我们面临的问题是如何构建验证集。请注意，在这个阶段，我们只知道数据点1,2,3,4,5,6，而数据点7和8对我们来说是未知的。
#正确的做法应该是，首先使用（4,5,6）预测出一个估计的第七个数据点，记作7#。然后，我们再使用（5,6,7#）来预测第八个数据点。这个过程被称为滚动预测。
#因此，划分的验证集，其实就第一行的数据可以用！

#建立模型和滚动预测
#决策树回归模型
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# 初始化决策树模型
tree_model = DecisionTreeRegressor()

# 定义参数网格
param_grid = {
    'max_depth': [None, 3, 5, 7, 9],
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 11)
}

# 初始化网格搜索
grid_search = GridSearchCV(tree_model, param_grid, cv=5, scoring='neg_mean_squared_error')
#scoring换成均方误差的负值。为什么使用负值呢？因为GridSearchCV的默认行为是认为分数（score）越大越好。
#而对于均方误差来说，值越小表示模型越好。通过使用负值，我们可以使GridSearchCV在优化时选择最小的均方误差。其他可选的还有：
#neg_mean_squared_error: 负均方误差
#neg_mean_absolute_error: 负平均绝对误差
#neg_median_absolute_error: 负中位绝对误差
#r2: R^2（决定系数）

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
best_params
#{'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 7}

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 使用最佳参数初始化决策树模型
best_tree_model = DecisionTreeRegressor(**best_params)

# 在训练集上训练模型
best_tree_model.fit(X_train, y_train)

#滚动预测
# 对于验证集，我们需要迭代地预测每一个数据点
y_validation_pred = []

for i in range(len(X_validation)):
    if i == 0:
        # 使用最后一个训练集的数据预测验证集的第一个数据点
        pred = best_tree_model.predict([X_validation.iloc[0]])
    else:
        # 使用前面预测出的数据构建新的特征，然后预测下一个数据点
        new_features = list(X_validation.iloc[i, 1:]) + [pred[0]]  # 将前面的特征向前移动，并使用上一次的预测作为最新的特征
        pred = best_tree_model.predict([new_features])
    y_validation_pred.append(pred[0])

y_validation_pred = np.array(y_validation_pred)

#模型评价
# 计算验证集上的MAE, MAPE, MSE和RMSE
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)
mae_validation, mape_validation, mse_validation, rmse_validation
#(0.012704805555555554, 0.7397855533427676, 0.00023204786595381854, 0.015233117407603032)
# 计算训练集上的MAE, MAPE, MSE和RMSE
y_train_pred = best_tree_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train, mape_train, mse_train, rmse_train
#(0.004381293602693602, 0.1919475518841738, 3.946459197306397e-05, 0.00628208500205656)
#观察模型拟合精度
#①ME：平均误差；
#②RMSE：均方根误差；
#③MAE：平均绝对误差，范围[0,+∞)，当预测值与真实值完全吻合时等于0，即完美模型；
#④MPE：平均百分比误差；
#⑤MAPE：平均绝对百分误差，范围[0,+∞)，MAPE 为0%表示完美模型，MAPE大于 100 %则表示劣质模型；
#⑥MASE：平均绝对标准化误差。