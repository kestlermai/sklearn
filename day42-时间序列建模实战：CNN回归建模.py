# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:30:31 2023

@author: maihuanzhuo
"""

#时间序列建模实战：CNN回归建模

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
#X_train = train_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']]
#返回的是Pandas DataFrame对象，
X_train = train_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']].values
#而加了values函数返回的是NumPy 数组
y_train = train_data['incidence'].values

X_validation = validation_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6']].values
y_validation = validation_data['incidence'].values


# 对于CNN，我们需要将输入数据重塑为3D格式 [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)
# 第一个维度表示样本的数量，即数据集中样本的个数。lag=6，即代表6个样本。
# 第二个维度表示时间步长，即每个样本中包含的时间步的数量或序列的长度。
# 第三个维度表示通道数，对于一维卷积神经网络，通常只有一个通道，因为输入数据是一维的。

'''
卷积神经网络（CNN）最初是为图像识别和处理而设计的，但它们已经被证明对于各种类型的序列数据，包括时间序列，也是有效的。

a）局部感受野:
-CNN的关键特点是它的局部感受野，这意味着每个卷积核只查看输入数据的一个小部分。
-对于时间序列，这意味着CNN可以捕获和学习模式中的短期依赖关系或周期性。
-这类似于在时间序列分析中使用滑动窗口来捕获短期模式。

（b）参数共享:
-在CNN中，卷积核的权重在输入的所有部分上都是共享的。
-这意味着网络可以在时间序列的任何位置都识别出相同的模式，增加了其泛化能力。

（c）多尺度特征捕获:
-通过使用多个卷积层和池化层，CNN能够在不同的时间尺度上捕获模式。
-这使得它们能够捕获长期和短期的时间序列依赖关系。

（d）堆叠结构:
多层的CNN结构使得网络可以学习时间序列中的复杂和抽象的模式。
例如，第一层可能会捕获简单的趋势或周期性，而更深层的网络可能会捕获更复杂的季节性模式或其他非线性关系。

（e）自动特征学习:
-传统的时间序列分析方法通常需要手动选择和构造特征。
-使用CNN，网络可以自动从原始数据中学习和提取相关特征，这通常导致更好的性能和更少的手工工作。

（f）时间序列的结构化特征:
-和图像数据一样，时间序列数据也具有结构性。例如，过去的观察结果通常影响未来的观察结果。
-CNN利用这种结构性，通过卷积操作从数据中提取局部和全局的时间模式。

总之，虽然CNN最初是为图像设计的，但它们在处理序列数据，特别是时间序列数据时，已经显示出了很强的潜力。
这是因为它们可以自动从数据中学习重要的特征，捕获多种尺度的模式，并适应时间序列中的短期和长期依赖关系。
'''
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

# 构建CNN模型
model = Sequential()
#添加一维卷积层
model.add(Conv1D(filters=64, #使用64个滤波器
                 kernel_size=2, #卷积核大小为2，即每次滑动2个联系的数据点
                 activation='relu', #激活函数引入非线性
                 input_shape=(X_train.shape[1], 1)))#input数据的形状
#添加一维池化层
model.add(MaxPooling1D(pool_size=2))#取每两个连续值的最大值来减少数据的维度
#添加展平层
model.add(Flatten())
#添加全链接层
model.add(Dense(50, activation='relu'))#50个神经元
#输出层为回归问题，而且是单步滚动预测，所以是1
model.add(Dense(1))

#定义优化器
from tensorflow.python.keras.optimizers import adam_v2
model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss='mse')

## 训练模型
history = model.fit(X_train, y_train, 
                    epochs=200, 
                    batch_size=32, 
                    validation_data=(X_validation, y_validation), 
                    verbose=0)

# 定义单步滚动预测函数
def rolling_forecast(model, initial_features, n_forecasts):
    forecasts = []
    current_features = initial_features.copy()
    #循环需要执行的滚动预测步数
    for i in range(n_forecasts):
        # 使用当前的特征进行预测
        forecast = model.predict(current_features.reshape(1, len(current_features), 1)).flatten()[0]#转化为3D格式
        #通过flatten()[0]将预测结果展平为一维数组，并提取第一个元素作为预测值
        forecasts.append(forecast)#添加到forecast列表中
        
        # 更新特征，用新的预测值替换最旧的特征
        current_features = np.roll(current_features, shift=-1)#np.roll函数将特征向左滚动一步，并将最旧的特征值替换为最新的预测值，以便在下一步预测中使用更新后的特征。
        current_features[-1] = forecast#current_features[-1]即最旧特征值，被替换成最新的预测值
    
    return np.array(forecasts)

# 使用训练集的最后6个数据点作为初始特征
initial_features = X_train[-1].flatten()

# 使用单步滚动预测方法预测验证集
y_validation_pred = rolling_forecast(model, initial_features, len(X_validation))#使用最近6个时间点数据去预测12个时间点数据

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 计算训练集上的MAE, MAPE, MSE 和 RMSE
mae_train = mean_absolute_error(y_train, model.predict(X_train).flatten())
mape_train = np.mean(np.abs((y_train - model.predict(X_train).flatten()) / y_train))
mse_train = mean_squared_error(y_train, model.predict(X_train).flatten())
rmse_train = np.sqrt(mse_train)

# 计算验证集上的MAE, MAPE, MSE 和 RMSE
mae_validation = mean_absolute_error(y_validation, y_validation_pred)
mape_validation = np.mean(np.abs((y_validation - y_validation_pred) / y_validation))
mse_validation = mean_squared_error(y_validation, y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

print("验证集：", mae_validation, mape_validation, mse_validation, rmse_validation)
# 验证集： 0.021895624911598865 1.6082488416268335 0.0007914206256882115 0.028132199090867595
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
# 训练集： 0.00922579876523051 0.4466329571205112 0.00017645801617652908 0.013283750079571999

#-----------------------------------------------多步滚动预测-----取均值

import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('data.csv')
data['time'] = pd.to_datetime(data['time'], format='%b-%y')

n = 6
m = 2

# 创建滞后期特征
for i in range(n, 0, -1):
    data[f'lag_{i}'] = data['incidence'].shift(n - i + 1)

data = data.dropna().reset_index(drop=True)

train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
validation_data = data[(data['time'] >= '2012-01-01') & (data['time'] <= '2012-12-31')]

#跟之前的代码不同，但思路是一样的
# 准备训练数据
X_train = []
y_train = []

#循环次数（用于生成训练样本的数量）=总的训练集长度-6个历史值-2个预测，+1是因为range不包含最后一位，但实际上要循环最后一位数
for i in range(len(train_data) - n - m + 1):
    X_train.append(train_data.iloc[i+n-1][[f'lag_{j}' for j in range(1, n+1)]].values)#同样，这里转换成Numpy数组
    y_train.append(train_data.iloc[i+n:i+n+m]['incidence'].values)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train.astype(np.float32)#astype函数将数据转换成单精度浮点数，32精度为7位小数，64精度（双精度）为15位小数
y_train = y_train.astype(np.float32)

# 为CNN准备数据
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)#转换成3d结构

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

# 构建CNN模型
model = Sequential()
#添加一维卷积层
model.add(Conv1D(filters=64, #使用64个滤波器
                 kernel_size=2, #卷积核大小为2，即每次滑动2个联系的数据点
                 activation='relu', #激活函数引入非线性
                 input_shape=(X_train.shape[1], 1)))#input数据的形状
#添加一维池化层
model.add(MaxPooling1D(pool_size=2))#取每两个连续值的最大值来减少数据的维度
#添加展平层
model.add(Flatten())
#添加全链接层
model.add(Dense(50, activation='relu'))#50个神经元
#输出层为回归问题，一次性预测m个值，所以是m
model.add(Dense(m))

#定义优化器
from tensorflow.python.keras.optimizers import adam_v2
model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

# for i in range(len(X_train) - m + 1):#在多步滚动预测中，每次预测出来都有m个值，而前m-1个预测结果都与上一次的预测结果存在重叠，因此就需要跳过m-1个时间步
#     pred = best_rf_model.predict([X_train.iloc[i]])
#     y_train_pred.extend(pred[0])

# # 重叠预测值取平均
# for i in range(1, m):
#     for j in range(len(y_train_pred) - i):
#         y_train_pred[j+i] = (y_train_pred[j+i] + y_train_pred[j]) / 2

#定义滚动函数
def cnn_rolling_forecast(data, model, n, m):
    y_pred = []
    #保留n个时间数据点作为历史数据
    for i in range(len(data) - n):
        input_data = data.iloc[i+n-1][[f'lag_{j}' for j in range(1, n+1)]].values.astype(np.float32).reshape(1, n, 1)#（一个样本，n为步长，一维通道）
        pred = model.predict(input_data)
        y_pred.extend(pred[0])
    #  重叠预测值取平均
    for i in range(1, m):
        for j in range(len(y_pred) - i):
            y_pred[j+i] = (y_pred[j+i] + y_pred[j]) / 2

    return np.array(y_pred)

#滚动预测
y_train_pred_cnn = cnn_rolling_forecast(train_data, model, n, m)[:len(y_train)]#保留与y_train集相同长度的结果
#仅保留与验证数据集长度减去历史时间步数相同长度（n=6）的预测结果。
y_validation_pred_cnn = cnn_rolling_forecast(validation_data, model, n, m)[:len(validation_data) - n]

from sklearn.metrics import mean_absolute_error, mean_squared_error
# Calculate performance metrics for train_data
#从第 n 个时间步开始，直到预测结果 y_train_pred_cnn 长度加上 n 个时间步为止，这里是90个样本，从6到89（不包含89）共83个值
mae_train = mean_absolute_error(train_data['incidence'].values[n:len(y_train_pred_cnn)+n], y_train_pred_cnn)
mape_train = np.mean(np.abs((train_data['incidence'].values[n:len(y_train_pred_cnn)+n] - y_train_pred_cnn) / train_data['incidence'].values[n:len(y_train_pred_cnn)+n]))
mse_train = mean_squared_error(train_data['incidence'].values[n:len(y_train_pred_cnn)+n], y_train_pred_cnn)
rmse_train = np.sqrt(mse_train)

# Calculate performance metrics for validation_data
mae_validation = mean_absolute_error(validation_data['incidence'].values[n:len(y_validation_pred_cnn)+n], y_validation_pred_cnn)
mape_validation = np.mean(np.abs((validation_data['incidence'].values[n:len(y_validation_pred_cnn)+n] - y_validation_pred_cnn) / validation_data['incidence'].values[n:len(y_validation_pred_cnn)+n]))
mse_validation = mean_squared_error(validation_data['incidence'].values[n:len(y_validation_pred_cnn)+n], y_validation_pred_cnn)
rmse_validation = np.sqrt(mse_validation)

print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.016809176324770893 1.1005987214213921 0.0005355304108647898 0.023141530002676783
print("验证集：", mae_validation, mape_validation, mse_validation, rmse_validation)
#验证集： 0.014963036192134025 1.496270066119133 0.00029670067408148594 0.017225001424716516

#----------------------------------------多步滚动预测----预测一半的数据（即删掉一半数据）

import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('data.csv')
data['time'] = pd.to_datetime(data['time'], format='%b-%y')

n = 6
m = 2

# 创建滞后期特征
for i in range(n, 0, -1):
    data[f'lag_{i}'] = data['incidence'].shift(n - i + 1)

data = data.dropna().reset_index(drop=True)

#划分训练集和验证集
train_data = data[(data['time'] >= '2004-01-01') & (data['time'] <= '2011-12-31')]
validation_data = data[(data['time'] >= '2012-01-01') & (data['time'] <= '2012-12-31')]

# 只对X_train、y_train、X_validation取奇数行
X_train = train_data[[f'lag_{i}' for i in range(1, n+1)]].iloc[::2].reset_index(drop=True).values
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Reshape for CNN

# 创建m个目标变量，同样通过values函数转换成NumPy数组
y_train_list = [train_data['incidence'].shift(-i) for i in range(m)]
y_train = pd.concat(y_train_list, axis=1)
y_train.columns = [f'target_{i+1}' for i in range(m)]
y_train = y_train.iloc[::2].reset_index(drop=True).dropna().values[:, 0]  # Only take the first column for simplicity

X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]].iloc[::2].reset_index(drop=True).values
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)  # Reshape for CNN

y_validation = validation_data['incidence'].values

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
# 构建CNN模型
model = Sequential()
#添加一维卷积层
model.add(Conv1D(filters=64, #使用64个滤波器
                 kernel_size=2, #卷积核大小为2，即每次滑动2个联系的数据点
                 activation='relu', #激活函数引入非线性
                 input_shape=(X_train.shape[1], 1)))#input数据的形状
#添加一维池化层
model.add(MaxPooling1D(pool_size=2))#取每两个连续值的最大值来减少数据的维度
#添加展平层
model.add(Flatten())
#添加全链接层
model.add(Dense(50, activation='relu'))#50个神经元
#输出层为回归问题，取奇数行进行预测，那么就是每个时间点预测出来的值都是独立的，所以取1
model.add(Dense(1))

#定义编译优化器
from tensorflow.python.keras.optimizers import adam_v2
model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

# Predict on validation set
y_validation_pred = model.predict(X_validation).flatten()

from sklearn.metrics import mean_absolute_error, mean_squared_error
# Compute metrics for validation set
mae_validation = mean_absolute_error(y_validation[:len(y_validation_pred)], y_validation_pred)#截断时间点，保证与验证集的时间点长度一致
mape_validation = np.mean(np.abs((y_validation[:len(y_validation_pred)] - y_validation_pred) / y_validation[:len(y_validation_pred)]))
mse_validation = mean_squared_error(y_validation[:len(y_validation_pred)], y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

# Predict on training set
y_train_pred = model.predict(X_train).flatten()

# Compute metrics for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train))
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

print("验证集：", mae_validation, mape_validation, mse_validation, rmse_validation)
#验证集： 0.018896390884419283 0.840648657734392 0.0008145250209843051 0.02853988474020708
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.009544734575125906 0.5201385006846897 0.0001748204249956296 0.013221967516055604


#-------------------------------------------多步滚动预测---构建m个CNN模型预测m个值，一个m值训练一个CNN模型

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

# 截断y_train使其与X_train的长度匹配，并转换成三维格式
for i in range(m):
    X_train_list[i] = X_train_list[i].iloc[:-(m-1)].values#这行代码将X_train_list中的每个元素（即特征数据）从末尾截断m-1行。例如，如果m=3，则截断最后2行。
    X_train_list[i] = X_train_list[i].reshape(X_train_list[i].shape[0], X_train_list[i].shape[1], 1)  # Reshape for CNN
    y_train_list[i] = y_train_list[i].iloc[:len(X_train_list[i])].values#这行代码确保标签数据的长度与特征数据的长度相匹配。
#综上所述，我们得到的X_train_list包含三个相同的输入集（A\B\C）；同样，y_train_list包含三个输出集（D\E\F），注意D\E\F的数据不一样。
#A和D用于训练模型一，B和E用于训练模型二，C和F用于训练模型三。

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.python.keras.optimizers import adam_v2

#构建CNN模型
models = []
for i in range(m):
    # Build CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_list[i].shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))#一个模型一直一个值，所以还是1
    #定义每个模型的编译优化器
    optimizer = adam_v2.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train_list[i], y_train_list[i], epochs=200, batch_size=32, verbose=0)
    models.append(model)


# 为了使validation_data的划分遵循上述的逻辑，我们首先需要确定其开始的时间点
# 这是在train_data最后一个时间点之后的第一个时间点
validation_start_time = train_data['time'].iloc[-1] + pd.DateOffset(months=1)
validation_data = data[data['time'] >= validation_start_time]
X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]].values
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)  # Reshape for CNN

y_validation_pred_list = [model.predict(X_validation) for model in models]
y_train_pred_list = [model.predict(X_train_list[i]) for i, model in enumerate(models)]

#对于m个预测结果进行拼接：
def concatenate_predictions(pred_list):
    concatenated = []
    for j in range(len(pred_list[0])):
        for i in range(m):
            concatenated.append(pred_list[i][j])
    return concatenated


y_validation_pred = np.array(concatenate_predictions(y_validation_pred_list))[:len(validation_data['incidence'])]
y_train_pred = np.array(concatenate_predictions(y_train_pred_list))[:len(train_data['incidence']) - m + 1]

#通过flatten函数扁平化转换成一维数据
y_validation_pred = y_validation_pred.flatten()
y_train_pred = y_train_pred.flatten()

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_validation = mean_absolute_error(validation_data['incidence'], y_validation_pred)
mape_validation = np.mean(np.abs((validation_data['incidence'] - y_validation_pred) / validation_data['incidence']))
mse_validation = mean_squared_error(validation_data['incidence'], y_validation_pred)
rmse_validation = np.sqrt(mse_validation)

mae_train = mean_absolute_error(train_data['incidence'][:-(m-1)], y_train_pred)
mape_train = np.mean(np.abs((train_data['incidence'][:-(m-1)] - y_train_pred) / train_data['incidence'][:-(m-1)]))
mse_train = mean_squared_error(train_data['incidence'][:-(m-1)], y_train_pred)
rmse_train = np.sqrt(mse_train)

print("验证集：", mae_validation, mape_validation, mse_validation, rmse_validation)
#验证集： 0.013390985294654966 1.5352366364844372 0.00034503802960929005 0.01857519931546604
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.02495269763741134 1.4155578377763118 0.0010576135670063618 0.0325209711879329
