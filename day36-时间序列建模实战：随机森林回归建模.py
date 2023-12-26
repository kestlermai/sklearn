# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 20:06:31 2023

@author: maihuanzhuo
"""

#day36-时间序列建模实战：随机森林回归建模
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#随机森林模型
"""
class sklearn.ensemble.RandomForestRegressor(n_estimators=100, *, #树的数量
                                             criterion='squared_error', #用于测量分裂质量的函数，
                                             #RandomForestRegressor 可选：{'mse', 'mae'}，默认为 'mse'。
                                             #RandomForestClassifier 可选：{'gini', 'entropy'}，默认为 'gini'。
                                             max_depth=None, #树的最大深度
                                             min_samples_split=2, #分裂内部节点所需的最小样本数
                                             min_samples_leaf=1, #叶节点所需的最小样本数
                                             min_weight_fraction_leaf=0.0, #叶节点所需的权重的最小加权总和
                                             max_features=1.0, #在寻找最佳分裂时考虑的特征数量
                                             max_leaf_nodes=None, #使用 max_depth 之前的最大叶子节点数
                                             min_impurity_decrease=0.0, #如果节点分裂会导致杂质的减少大于或等于该值，则该节点将被分裂。
                                             bootstrap=True, #是否使用 bootstrap 样本进行建树
                                             oob_score=False, #是否使用 out-of-bag 样本来估计泛化准确度
                                             n_jobs=None, #并行运行的任务数
                                             random_state=None, #用于控制随机性的种子
                                             verbose=0, #输出过程
                                             warm_start=False, #设置为 True 时，重用前一个调用的解决方案来适应并为森林添加更多的估计器
                                             #特定于 RandomForestClassifier 的参数
                                             ccp_alpha=0.0, #用于最小化成本复杂性修剪的复杂性参数。具有最大成本复杂性的树会被修剪
                                             max_samples=None)#从 X 中抽取的样本数量，用于训练每个基本估计器。
ctrl +1 批量注释
"""
# 初始化随机森林模型
rf_model = RandomForestRegressor()

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 7],
}

# 初始化网格搜索
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
best_params
#{'max_depth': 5, 'n_estimators': 50}

# 使用最佳参数初始化随机森林模型
best_rf_model = RandomForestRegressor(**best_params)

# 在训练集上训练模型
best_rf_model.fit(X_train, y_train)

# 对于验证集，我们需要迭代地预测每一个数据点
y_validation_pred = []

for i in range(len(X_validation)):
    if i == 0:
        pred = best_rf_model.predict([X_validation.iloc[0]])#使用验证集第一个时间步经过RF模型预测（由于没有历史预测值）
    else:
        new_features = list(X_validation.iloc[i, 1:]) + [pred[0]] #模型每次都利用前一步的预测结果作为输入特征来预测下一步的值
        pred = best_rf_model.predict([new_features])
    y_validation_pred.append(pred[0])

y_validation_pred = np.array(y_validation_pred)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 计算验证集上的MAE, MAPE, MSE和RMSE
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

# 计算训练集上的MAE, MAPE, MSE和RMSE
y_train_pred = best_rf_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

print("Train Metrics:", mae_train, mape_train, mse_train, rmse_train)
#Train Metrics: 0.005385700738515836 0.31073203259557153 4.7365336138839146e-05 0.006882247898676649
print("Validation Metrics:", mae_validation, mape_validation, mse_validation, rmse_validation)
#Validation Metrics: 0.018000528403323533 1.7743413925715188 0.0006482773779322544 0.025461291756944587


#多步滚动预测
#同样用前面6个预测后面2个值

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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
#构建RF模型
rf_model = RandomForestRegressor()

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 3, 5, 7, 9],

}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_params
#{'max_depth': 5, 'n_estimators': 50}
best_rf_model = RandomForestRegressor(**best_params)
best_rf_model.fit(X_train, y_train)

# 预测验证集
y_validation_pred = []

for i in range(len(X_validation) - m + 1):
    pred = best_rf_model.predict([X_validation.iloc[i]])
    y_validation_pred.extend(pred[0])

# 重叠预测值取平均
for i in range(1, m):
    for j in range(len(y_validation_pred) - i):
        y_validation_pred[j+i] = (y_validation_pred[j+i] + y_validation_pred[j]) / 2

y_validation_pred = np.array(y_validation_pred)[:len(y_validation)]

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

print(mae_validation, mape_validation, mse_validation, rmse_validation)
#0.017047243568303647 1.1789898596343764 0.0005429669541887333 0.023301651318924445

# 拟合训练集
y_train_pred = []

for i in range(len(X_train) - m + 1):#在多步滚动预测中，每次预测出来都有m个值，而前m-1个预测结果都与上一次的预测结果存在重叠，因此就需要跳过m-1个时间步
    pred = best_rf_model.predict([X_train.iloc[i]])
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
#0.024849709861624943 1.1650660983338892 0.0010196669376722175 0.03193222412661256

#多步滚动预测v2-----删除一半的输入数据集

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

# 只对X_train、y_train、X_validation取奇数行，也就是我们说的删除一半的输入数据集。
X_train = train_data[[f'lag_{i}' for i in range(1, n+1)]].iloc[::2].reset_index(drop=True)

# 创建m个目标变量
y_train_list = [train_data['incidence'].shift(-i) for i in range(m)]
y_train = pd.concat(y_train_list, axis=1)
y_train.columns = [f'target_{i+1}' for i in range(m)]
y_train = y_train.iloc[::2].reset_index(drop=True).dropna()
X_train = X_train.head(len(y_train))

#对X_validation取奇数行
X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]].iloc[::2].reset_index(drop=True)
y_validation = validation_data['incidence']

#构建RF model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
rf_model = RandomForestRegressor()

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 3, 5, 7, 9],

}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_params
#{'max_depth': 7, 'n_estimators': 100}
best_rf_model = RandomForestRegressor(**best_params)
best_rf_model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 预测验证集
y_validation_pred = []

for i in range(len(X_validation)):
    pred = best_rf_model.predict([X_validation.iloc[i]])
    y_validation_pred.extend(pred[0])

y_validation_pred = np.array(y_validation_pred)[:len(y_validation)]

mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

print(mae_validation, mape_validation, mse_validation, rmse_validation)
#0.009804939791666661 0.8736942618940989 0.00012465933206774955 0.011165094360002048

# 预测训练集
y_train_pred = []

for i in range(len(X_train)):
    pred = best_rf_model.predict([X_train.iloc[i]])
    y_train_pred.extend(pred[0])

y_train_pred = np.array(y_train_pred)[:y_train.shape[0]]

mae_train = mean_absolute_error(y_train.iloc[:, 0], y_train_pred)
mape_train = np.mean(np.abs((y_train.iloc[:, 0] - y_train_pred) / y_train.iloc[:, 0]))
mse_train = mean_squared_error(y_train.iloc[:, 0], y_train_pred)
rmse_train = np.sqrt(mse_train)

print(mae_train, mape_train, mse_train, rmse_train)
#0.026989329666666666 1.1737411043970203 0.0014539713940371012 0.03813097683035541

#多步滚动预测-vol. 3-----构建多个模型去分别预测m个值
#同样都是构建RF model

import pandas as pd
import numpy as np
# 数据读取和预处理
data = pd.read_csv('data.csv')#用于构建滞后期特征x
data_y = pd.read_csv('data.csv')#用于构建响应变量y，一个模型预测一个值，那么，对于第一个模型（i=0），我们从第n个数据点开始选择标签。
#对于第二个模型（i=1），我们从第n+1个数据点开始选择标签。对于第三个模型（i=2），我们从第n+2个数据点开始选择标签。
#所以我们导入一份相同数据用于构建响应变量y
data['time'] = pd.to_datetime(data['time'], format='%b-%y')
data_y['time'] = pd.to_datetime(data_y['time'], format='%b-%y')

#创建滞后期特征
n = 6

for i in range(n, 0, -1):
    data[f'lag_{i}'] = data['incidence'].shift(n - i + 1)

data = data.dropna().reset_index(drop=True)

#划分训练集x_train
train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
X_train = train_data[[f'lag_{i}' for i in range(1, n+1)]]

m = 3

X_train_list = []
y_train_list = []

#因为每个y_train的时间点对应不一样的x_train
for i in range(m):
    X_temp = X_train#但每个模型的x_train都是一样的所以X_temp = X_train
    y_temp = data_y['incidence'].iloc[n + i:len(data_y) - m + 1 + i]#而y_train都是分别对应m个模型的时间点
    X_train_list.append(X_temp)
    y_train_list.append(y_temp)

#原始数据中共有108个时间点，而去除6个滞后期特征剩102个时间点，然后拆分为90个训练集（6个滞后特征），12个验证集。
#因此x_train为90个时间点，y_train应该对应x_train一样也是90个时间点
#但现在y_train为100个时间点，因为在这里我们循环了m=3次，即迭代了3次(0,1,2)，即(6：105)=100, (7:106)=100, (8:107)=100
#如果是m=4次，那么y_train为99个时间点，即6:104（108-4）=99

#修剪x_train跟y_train为相同长度
for i in range(m):
    X_train_list[i] = X_train_list[i].iloc[:-(m-1)]#删除前m-1个时间点，3-1=2，那么就是90-2=88个时间点，
    #意思是最多用之前0:88个时间点去预测3个值（包括第88个），确保每个时间点都有足够的历史时间点
    y_train_list[i] = y_train_list[i].iloc[:len(X_train_list[i])]
#x_train为88个，y_train也为88个

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# 模型训练
rf_model = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 3, 5, 7, 9],

}

best_rf_models = []

for i in range(m):
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_list[i], y_train_list[i])
    best_params = grid_search.best_params_
    print(f"Best Parameters for Model {i + 1}: {best_params}")
    best_rf_model = RandomForestRegressor(**grid_search.best_params_)
    best_rf_model.fit(X_train_list[i], y_train_list[i])
    best_rf_models.append(best_rf_model)
# Best Parameters for Model 1: {'max_depth': 3, 'n_estimators': 50}
# Best Parameters for Model 2: {'max_depth': 3, 'n_estimators': 150}
# Best Parameters for Model 3: {'max_depth': 5, 'n_estimators': 150}

validation_start_time = train_data['time'].iloc[-1] + pd.DateOffset(months=1)
validation_data = data[data['time'] >= validation_start_time]

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]]#提取验证集滞后特征用于RF model生成验证集的预测值
y_validation_pred_list = [model.predict(X_validation) for model in best_rf_models]
y_train_pred_list = [model.predict(X_train_list[i]) for i, model in enumerate(best_rf_models)]

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
#验证集： 0.01615674127931087 1.191379050734296 0.0004498076768535742 0.021208669851114526

mae_train = mean_absolute_error(train_data['incidence'][:-(m-1)], y_train_pred)
mape_train = np.mean(np.abs((train_data['incidence'][:-(m-1)] - y_train_pred) / train_data['incidence'][:-(m-1)]))
mse_train = mean_squared_error(train_data['incidence'][:-(m-1)], y_train_pred)
rmse_train = np.sqrt(mse_train)
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.025020566749119955 1.3649100288164453 0.0010719325401358202 0.03274038087951666

