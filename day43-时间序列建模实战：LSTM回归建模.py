# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:37:31 2023

@author: maihuanzhuo
"""

# 时间序列建模实战：LSTM回归建模
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


# 对于LSTM，我们需要将输入数据重塑为3D格式 [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)
# 第一个维度表示样本的数量，即数据集中样本的个数。lag=6，即代表6个样本。
# 第二个维度表示时间步长，即每个样本中包含的时间步的数量或序列的长度。
# 第三个维度表示通道数，对于一维卷积神经网络，通常只有一个通道，因为输入数据是一维的。

'''
首先，Recurrent Neural Network (RNN)--循环神经网络模型（用来分析序列数据）

在全连接神经网络的基础上增加了前后时序上的关系，可以更好地处理比如机器翻译等的与时序相关的问题。

在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的。

一个典型的 RNN 网络架构包含一个输入，一个输出和一个神经网络单元 。Xt---A---ht。

而中间的这个RNN神经网络单元可以存在一个循环或者是回路 A---A---A---A；如此类推到下一个神经元。（每个神经元之间存在权重）

这就是说明第一个神经元的特征信息可以通过一定的权重一直保留到之后的神经元当中。（RNN的核心：参数共享）

即：上一个时刻的网络状态将会作用于（影响）到下一个时刻的网络状态。

同时，RNN要求每个时刻都要求有输入，但不要求有输出。

RNN向传播算法得到的预测值与真实值构建损失函数（分类：交叉熵损失 Cross Entropy Loss；回归：均方误差 MSE）
使用梯度下降最小化损失函数（即不断地更新参数）

后向传播算法的核心思想就是采用梯度下降算法进行一步步的迭代，直到得到最终需要的参数U、V、W、b、c，反向传播算法也被称为BPTT(back-propagation through time)。

但是由于激活函数的的原因（sigmoid函数和tanh函数：一堆小数在做乘法，结果就是越乘越小。）或者是相乘>1的数-----梯度爆炸
sigmoid函数的导数范围是(0,0.25]，tanH函数的导数范围是(0,1]，他们的导数最大都不大于1。
                                     
因此，随着序列越来越长，参数无法有效更新，就会出现梯度消失或者是梯度爆炸

那么RNN就是不能使用太长的时间序列数据，因此就出现长短期记忆（LSTM）和门控循环单元（Gated Recurrent Unit，GRU）；帮助学习长期历史数据


（1）LSTM简介：https://www.bilibili.com/video/BV1YK411F7Tg/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2cae71d4e74b72ef59e161b64db36f18

 LSTM (Long Short-Term Memory) 是一种特殊的RNN（递归神经网络）结构，由Hochreiter和Schmidhuber在1997年首次提出。
 LSTM 被设计出来是为了避免长序列在训练过程中的长期依赖问题，这是传统 RNNs 所普遍遇到问题。

（a）LSTM 的主要特点：

（a1）三个门结构：LSTM 包含三个门结构：输入门、遗忘门和输出门。这些门决定了信息如何进入、被存储或被遗忘，以及如何输出。

（a2）记忆细胞：LSTM的核心是称为记忆细胞的结构。它可以保留、修改或访问的内部状态。通过门结构，模型可以学会在记忆细胞中何时存储、忘记或检索信息。

（a3）长期依赖问题：LSTM特别擅长学习、存储和使用长期信息，从而避免了传统RNN在长序列上的梯度消失问题。

（b）为什么LSTM适合时间序列建模：

（b1）序列数据的特性：时间序列数据具有顺序性，先前的数据点可能会影响后面的数据点。LSTM设计之初就是为了处理带有时间间隔、延迟和长期依赖关系的序列数据。

（b2）长期依赖：在时间序列分析中，某个事件可能会受到很早之前事件的影响。传统的RNNs由于梯度消失的问题，很难捕捉这些长期依赖关系。
 但是，LSTM结构可以有效地处理这种依赖关系。

（b3）记忆细胞：对于时间序列预测，能够记住过去的信息是至关重要的。LSTM的记忆细胞可以为模型提供这种存储和检索长期信息的能力。

（b4）灵活性：LSTM模型可以与其他神经网络结构（如CNN）结合，用于更复杂的时间序列任务，例如多变量时间序列或序列生成。

综上所述，由于LSTM的设计和特性，它非常适合时间序列建模，尤其是当数据具有长期依赖关系时。

看笔记比较清楚


心血来潮，这里也去了解一下门控循环单元（Gated Recurrent Unit，GRU）

同样，GRU也是RNN的变种，通过控制门的方式去解决RNN存在的梯度消失和爆炸的问题

但是GRU与LSTM相比，结构相对简单，GRU只有两个门：更新门（update gate）和重置门（reset gate）

更新门通过sigmoid函数*Zt（进入当期信息比例）*历史信息Ht-1====得到更新的历史信息

重置门同样通过sigmoid函数rt（上期信息遗忘比例）*历史信息Ht-1+【Xt（当前新信息）*Wx（权重）】===然后通过tanH函数求导，与（1-Zt）xt相乘得到新增历史部分

然后更新后的历史信息+新增的历史部分信息===当期Ht


最后说一下LSTM和GRU的区别：

1：GRU参数比LSTM少，所以容易收敛。 数据集大的情况下，LSTM的表达性能还是比GRU好。

2：在一般数据集上 GRU和LSTM的性能差不多

3：
从结构上来说，GRU只有两个门（update和reset），GRU直接将hidden state 传给下一个单元，

LSTM有三个门（forget，input，output），GRU直接将hidden state 传给下一个单元，而LSTM则用memory cell 把hidden state 包装起来

'''

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers, models, optimizers
from tensorflow.python.keras.optimizers import adam_v2

# 构建LSTM回归模型，当然如果想用GRU模型那只需要把layers.LSTM改成layers.GRU即可
# x = layers.GRU(50, return_sequences=True)(input_layer)
# x = layers.GRU(25, return_sequences=False)(input_layer)
input_layer = layers.Input(shape=(X_train.shape[1], 1))
#定义LSTM层
x = layers.LSTM(50, return_sequences=True)(input_layer)#设置50个LSTM单元学习数据特征，类似CNN中的神经元；该LSTM层返回完整的输出序列
x = layers.LSTM(25, return_sequences=False)(x)#设置25个LSTM单元提取前面一层的数据特征，然后返回输出
x = layers.Dropout(0.1)(x)#drop率0.1，可以往上调？
x = layers.Dense(25, activation='relu')(x)#全连接层设置25个神经元，通过RELU函数添加非线性
x = layers.Dropout(0.1)(x)
output_layer = layers.Dense(1)(x)#单步预测，回归问题，输出一个值

model = models.Model(inputs=input_layer, outputs=output_layer)

#定义编译器
model.compile(optimizer=adam_v2.Adam(learning_rate=0.001), loss='mse')#过拟合可以调小学习率

# 训练模型
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_validation, y_validation), verbose=0)

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
y_validation_pred = rolling_forecast(model, initial_features, len(X_validation))

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
# 验证集： 0.021087161858975894 1.967723848708366 0.0007016844119404748 0.02648932637762755
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
# 训练集： 0.014178412810441523 0.7428782438723657 0.00037049141678083833 0.019248153594068142

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
from tensorflow.python.keras import layers, models, optimizers

# 构建LSTM回归模型
input_layer = layers.Input(shape=(X_train.shape[1], 1))
#定义LSTM层，鸡哥设置第一个LSTM层为64，第二个为32，而且少一个drop层，然后全连接设置50，再drop层
# x = LSTM(64, return_sequences=True)(inputs)
# x = LSTM(32)(x)
# x = Dense(50, activation='relu')(x)
# x = Dropout(0.1)(x)
# outputs = Dense(m)(x)
x = layers.LSTM(50, return_sequences=True)(input_layer)
x = layers.LSTM(25, return_sequences=False)(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(25, activation='relu')(x)
x = layers.Dropout(0.1)(x)
output_layer = layers.Dense(m)(x)##输出层为回归问题，一次性预测m个值，所以是m

model = models.Model(inputs=input_layer, outputs=output_layer)

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
def lstm_rolling_forecast(data, model, n, m):
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
y_train_pred_lstm = lstm_rolling_forecast(train_data, model, n, m)[:len(y_train)]#保留与y_train集相同长度的结果
#仅保留与验证数据集长度减去历史时间步数相同长度（n=6）的预测结果。
y_validation_pred_lstm = lstm_rolling_forecast(validation_data, model, n, m)[:len(validation_data) - n]

from sklearn.metrics import mean_absolute_error, mean_squared_error
# Calculate performance metrics for train_data
#从第 n 个时间步开始，直到预测结果 y_train_pred_cnn 长度加上 n 个时间步为止，这里是90个样本，从6到89（不包含89）共83个值
mae_train = mean_absolute_error(train_data['incidence'].values[n:len(y_train_pred_lstm)+n], y_train_pred_lstm)
mape_train = np.mean(np.abs((train_data['incidence'].values[n:len(y_train_pred_lstm)+n] - y_train_pred_lstm) / train_data['incidence'].values[n:len(y_train_pred_lstm)+n]))
mse_train = mean_squared_error(train_data['incidence'].values[n:len(y_train_pred_lstm)+n], y_train_pred_lstm)
rmse_train = np.sqrt(mse_train)

# Calculate performance metrics for validation_data
mae_validation = mean_absolute_error(validation_data['incidence'].values[n:len(y_validation_pred_lstm)+n], y_validation_pred_lstm)
mape_validation = np.mean(np.abs((validation_data['incidence'].values[n:len(y_validation_pred_lstm)+n] - y_validation_pred_lstm) / validation_data['incidence'].values[n:len(y_validation_pred_lstm)+n]))
mse_validation = mean_squared_error(validation_data['incidence'].values[n:len(y_validation_pred_lstm)+n], y_validation_pred_lstm)
rmse_validation = np.sqrt(mse_validation)

print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.017085006167747913 1.0363990839264887 0.0005408132391940076 0.02325539161558041
print("验证集：", mae_validation, mape_validation, mse_validation, rmse_validation)
#验证集： 0.020805263176423807 2.6333576545250263 0.00047279622107590725 0.021743877783778755

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
from tensorflow.python.keras import layers, models, optimizers
from tensorflow.python.keras.optimizers import adam_v2

# 构建LSTM回归模型
input_layer = layers.Input(shape=(X_train.shape[1], 1))
#定义LSTM层
x = layers.LSTM(50, return_sequences=True)(input_layer)#设置50个LSTM单元学习数据特征，类似CNN中的神经元；该LSTM层返回完整的输出序列
x = layers.LSTM(25, return_sequences=False)(x)#设置25个LSTM单元提取前面一层的数据特征，然后返回输出
x = layers.Dropout(0.1)(x)#drop率0.1，可以往上调？
x = layers.Dense(25, activation='relu')(x)#全连接层设置25个神经元，通过RELU函数添加非线性
x = layers.Dropout(0.1)(x)
output_layer = layers.Dense(1)(x)#单步预测，回归问题，输出一个值
#输出层为回归问题，取奇数行进行预测，那么就是每个时间点预测出来的值都是独立的，所以取1

model = models.Model(inputs=input_layer, outputs=output_layer)

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
#验证集： 0.015515484451651572 0.8166105816683936 0.00026044417039795316 0.016138282758644217
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.021336239005724587 1.398917198443779 0.0007198748162981162 0.026830482968036865


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
from tensorflow.python.keras import layers, models, optimizers
from tensorflow.python.keras.optimizers import adam_v2

#构建LSTM模型
models_list = []#这里要改一下，不然跟models函数重名
for i in range(m):
    input_layer = layers.Input(shape=(X_train.shape[1], 1))
    #定义LSTM层
    x = layers.LSTM(50, return_sequences=True)(input_layer)#设置50个LSTM单元学习数据特征，类似CNN中的神经元；该LSTM层返回完整的输出序列
    x = layers.LSTM(25, return_sequences=False)(x)#设置25个LSTM单元提取前面一层的数据特征，然后返回输出
    x = layers.Dropout(0.1)(x)#drop率0.1，可以往上调？
    x = layers.Dense(25, activation='relu')(x)#全连接层设置25个神经元，通过RELU函数添加非线性
    x = layers.Dropout(0.1)(x)
    output_layer = layers.Dense(1)(x)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    #定义每个模型的编译优化器
    optimizer = adam_v2.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train_list[i], y_train_list[i], epochs=200, batch_size=32, verbose=0)
    models_list.append(model)#这里记得也要改


# 为了使validation_data的划分遵循上述的逻辑，我们首先需要确定其开始的时间点
# 这是在train_data最后一个时间点之后的第一个时间点
validation_start_time = train_data['time'].iloc[-1] + pd.DateOffset(months=1)
validation_data = data[data['time'] >= validation_start_time]
X_validation = validation_data[[f'lag_{i}' for i in range(1, n+1)]].values
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)  # Reshape for CNN

y_validation_pred_list = [model.predict(X_validation) for model in models_list]#这里也要改
y_train_pred_list = [model.predict(X_train_list[i]) for i, model in enumerate(models_list)]#这里也要改

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
#验证集： 0.01643475600279868 1.4455691251211178 0.00044976067454506635 0.0212075617303137
print("训练集：", mae_train, mape_train, mse_train, rmse_train)
#训练集： 0.021074589821723375 1.194225783492586 0.0007557803368816552 0.02749145934434284

# 鸡哥的代码中单步预测用的是多层LSTM，在后续进行多步预测中，逐步减少层数，变成单一层LSTM，对比看到性能有所提升
# 这意味着多层LSTM可能存在梯度消失和梯度爆炸问题，过拟合（数据量不足问题），某些时间序列并不需要更深的网络结构，反而可能单一层LSTM性能是最好的。
