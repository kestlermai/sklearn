# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:24:02 2023

@author: maihuanzhuo
"""

 #时间序列建模实战：多因素预测---ARIMAX
 #多变量时间序列预测模型（ARIMAX）
 
import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模/时间序列建模实战：多因素预测---ARIMAX') ##修改路径
 
 #读取数据（风速以及其他额外变量）
import pandas as pd 
data = pd.read_csv('wind_dataset.csv',
                   index_col=0, #指定数据第一列作为数据框的第一列
                   parse_dates=True)#将日期时间字符串解析为日期时间对象

#绘制时间序列图
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 7))
data['WIND'].plot(title='WIND Time Series Plot')
plt.ylabel('WIND')
plt.xlabel('Data')
plt.grid(True)
plt.tight_layout()
plt.show()

#相关性分析
#这一步，有点类似特征工程。
#这七个额外变量，哪一个对于风速的预测起到正向作用呢？是需要筛选的，大力不一定出奇迹，还得使用巧劲哈。


# 计算WIND与其他变量之间的相关性
correlations = data.corr()['WIND'].drop('WIND')
correlations
#IND       -0.038578
#RAIN       0.120876
#IND.1      0.070512
#T.MAX     -0.242559
#IND.2      0.047860
#T.MIN     -0.093014
#T.MIN.G    0.012823

#绘制相关性热图
import seaborn as sns 
corr = data.corr()
plt.figure(figsize = (8,6))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True, fmt='.2f', linewidths=.30)
plt.title('Correlation of WIND Features', y =1.05,  size=15)
pos, textvals = plt.yticks()
plt.yticks(pos,('IND','RAIN','IND.1','T.MAX','IND.2','T.MIN','T.MIN.G'), 
    rotation=0, fontsize="10", va="center")
#我们可以看到T.MAX与风速具有中度的负相关性，而RAIN与风速具有较低的正相关性。其他变量与风速的相关性较弱。

#同时我们需要考虑时间序列的滞后性问题，即今天的某个变量因素在未来1,2,3,..,n天内会影响结局变化。
#通过计算结局跟变量的滞后值的相关性，确定在哪个滞后值lag下与结局之间的相关性最强。

# 设置最大滞后期数为7，并包括0（表示当天）
max_lag_including_today = 7
exog_variables = {'IND','RAIN','IND.1','T.MAX','IND.2','T.MIN','T.MIN.G'}
correlation_results_0_to_7 = {}
for lag in range(0, max_lag_including_today + 1):
    lagged_correlations = {}
    for column in exog_variables:
        column_name = f'{column}_lag_{lag}'
        data[column_name] = data[column].shift(lag)
        correlation = data['WIND'].corr(data[column_name])
        lagged_correlations[column] = correlation
    correlation_results_0_to_7[lag] = lagged_correlations
correlation_df_0_to_7 = pd.DataFrame(correlation_results_0_to_7).T
correlation_df_0_to_7
#       RAIN     T.MAX   T.MIN.G       IND     IND.1     IND.2     T.MIN
#0  0.120876 -0.242559  0.012823 -0.038578  0.070512  0.047860 -0.093014
#1  0.158439 -0.225210 -0.136349 -0.056325  0.072248  0.071343 -0.189927
#2  0.063843 -0.247308 -0.168475 -0.027037  0.070012  0.079187 -0.205467
#3  0.036880 -0.254227 -0.184317 -0.026205  0.069726  0.083817 -0.209190
#4  0.034355 -0.253187 -0.180456 -0.032923  0.066721  0.084730 -0.207173
#5  0.027268 -0.256108 -0.175098 -0.030828  0.064991  0.078578 -0.197822
#6  0.017677 -0.250267 -0.161606 -0.017973  0.063408  0.071776 -0.186589
#7  0.009850 -0.250619 -0.162920 -0.022138  0.062131  0.071805 -0.197446
correlation_df_0_to_7.to_csv('correlation_df_0_to_7.csv')#导出滞后期结果

#（1）RAIN在当天和滞后1天还不错，在之后就不行了；
#（2）T.MAX不管滞后多久，依旧相关；
#（3）T.MIN滞后1天之后，开始展现出相关。T.MIN.G同样

#综上，我们尝试使用RAIN、T.MAX、T.MIN（lag3）和T.MIN.G（lag3）建立多因素预测模型。

#重新导入数据
data = pd.read_csv('wind_dataset.csv', index_col=0, parse_dates=True)

#创建滞后3天的特征
data['T.MIN_lag_3'] = data['T.MIN'].shift(3)
data['T.MIN.G_lag_3'] = data['T.MIN.G'].shift(3)

#分割训练集和测试集
train = data.iloc[:-30]  # 使用数据集中除最后30天的部分作为训练集
test = data.iloc[-30:]  # 使用数据集中最后30天的部分作为测试集

#定义外部解释变量（训练集和测试集）
exog_train = train[['RAIN', 'T.MAX', 'T.MIN_lag_3', 'T.MIN.G_lag_3']].dropna()#去除缺失值的行
exog_test = test[['RAIN', 'T.MAX', 'T.MIN_lag_3', 'T.MIN.G_lag_3']].dropna()

#在拟合模型之前，我个人觉得需要先做ADF检验跟格兰杰因果关系检验（鸡哥说格兰杰因果关系检验一般在计量经济学才用，可以不做）
#ADF
from statsmodels.tsa.stattools import adfuller
def augmented_dickey_fuller_statistics(time_series):
    result = adfuller(time_series.values)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
print('Augmented Dickey-Fuller Test: WIND Time Series')
augmented_dickey_fuller_statistics(data['WIND'])#p-value: 0.000000
print('Augmented Dickey-Fuller Test: RAIN Time Series')
augmented_dickey_fuller_statistics(exog_train['RAIN'])#p-value: 0.000000
print('Augmented Dickey-Fuller Test: T.MAX Time Series')
augmented_dickey_fuller_statistics(exog_train['T.MAX'])#p-value: 0.000000
print('Augmented Dickey-Fuller Test: T.MIN_lag_3 Time Series')
augmented_dickey_fuller_statistics(exog_train['T.MIN_lag_3'])#p-value: 0.000000
print('Augmented Dickey-Fuller Test: T.MIN.G_lag_3 Time Series')
augmented_dickey_fuller_statistics(exog_train['T.MIN.G_lag_3'])#p-value: 0.000000
#全部小于0.05

#格兰杰因果检验
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
maxlag = 3
test = 'ssr-chi2test'#卡方统计量和p值的元组，用于检验因变量的滞后变量是否有预测能力，同时考虑到了自变量的影响（与params_ftest相同）
#定义函数用来计算格兰杰因果关系矩阵
def grangers_causality_matrix(data, variables, test = 'ssr_chi2test', verbose=False):
    # 创建一个空的DataFrame，用于存储格兰杰因果关系检验的结果
    dataset = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    # 循环遍历所有可能的变量组合，计算格兰杰因果关系检验
    for c in dataset.columns:
        for r in dataset.index:
            test_result = grangercausalitytests(data[[r,c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            # # 找到最小的p值作为因果关系检验的结果
            min_p_value = np.min(p_values)
            dataset.loc[r,c] = min_p_value
    # 重命名列和索引以反映变量之间的因果关系
    dataset.columns = [var + '_x' for var in variables]
    dataset.index = [var + '_y' for var in variables]
    
    return dataset

#定义一个新的数据集用于格兰杰检验
data = data.dropna()
#拆分数据
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]
X_train = train[['WIND', 'RAIN', 'T.MAX', 'T.MIN_lag_3', 'T.MIN.G_lag_3']]

X_train_transformed = X_train.diff().dropna()

grangers_causality_matrix(X_train_transformed, variables = X_train_transformed.columns)
#                 WIND_x  RAIN_x  T.MAX_x  T.MIN_lag_3_x  T.MIN.G_lag_3_x
#WIND_y           1.0000     0.0   0.0000         0.0022           0.0001
#RAIN_y           0.0071     1.0   0.0194         0.0883           0.0290
#T.MAX_y          0.0000     0.0   1.0000         0.0009           0.0035
#T.MIN_lag_3_y    0.0000     0.0   0.0000         1.0000           0.0014
#T.MIN.G_lag_3_y  0.0000     0.0   0.0000         0.0000           1.0000

#RAIN_x  T.MAX_x  T.MIN_lag_3_x  T.MIN.G_lag_3_x与WIND_y存在因果关系（p<0.05）

#重新导入数据
data = pd.read_csv('wind_dataset.csv', index_col=0, parse_dates=True)

#创建滞后3天的特征
data['T.MIN_lag_3'] = data['T.MIN'].shift(3)
data['T.MIN.G_lag_3'] = data['T.MIN.G'].shift(3)

#分割训练集和测试集
train = data.iloc[:-30]  # 使用数据集中除最后30天的部分作为训练集
test = data.iloc[-30:]  # 使用数据集中最后30天的部分作为测试集

#定义外部解释变量（训练集和测试集）
exog_train = train[['RAIN', 'T.MAX', 'T.MIN_lag_3', 'T.MIN.G_lag_3']].dropna()#去除缺失值的行
exog_test = test[['RAIN', 'T.MAX', 'T.MIN_lag_3', 'T.MIN.G_lag_3']].dropna()

#定义因变量
endog_train = train.loc[exog_train.index, 'WIND']
endog_test = test.loc[exog_test.index, 'WIND']

import statsmodels.api as sm
# 设置参数范围
p_values = range(0, 2)
d_values = range(0, 1)
q_values = range(0, 2)

#基于AIC值、SBC值和模型参数的统计学意义来选择最优模型，一般根据AIC或BIC最小原则选择较优模型
# 用于存储最优模型的信息
best_model_info = {
    "AIC": float('inf'),
    "SBC": float('inf'),
    "R_squared": float('-inf'),
    "order": None,
    "model": None
}
# 遍历参数，寻找最优模型
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                # 构建ARIMAX模型
                model = sm.tsa.ARIMA(endog=endog_train, exog=exog_train, order=(p, d, q))
                # 拟合模型
                model_fit = model.fit(disp=False)
                # 计算AIC和SBC
                AIC = model_fit.aic
                SBC = model_fit.bic
                # 检查p值
                p_values_summary = model_fit.pvalues
                if all(p_value < 0.05 for p_value in p_values_summary):
                    # 检查是否是最优模型
                    if AIC < best_model_info["AIC"] and SBC < best_model_info["SBC"]:
                        best_model_info["AIC"] = AIC
                        best_model_info["SBC"] = SBC
                        best_model_info["order"] = (p, d, q)
                        best_model_info["model"] = model_fit
            except:
                continue

# 打印最优模型信息
best_model_info["order"], best_model_info["AIC"], best_model_info["SBC"]
#(None, inf, inf)，没有找到合适的参数
# 获取最优模型各个参数的p值
p_values_of_best_model = best_model_info["model"].pvalues

# 打印各个参数的p值
print("最优模型各个参数的p值：")
for param, p_value in p_values_of_best_model.items():
    print(f"{param}: {p_value}") 
#没有找到合适的参数

#拟合ARIMAX模型
#由于ARIMA和ARIMAX模型的参数（如差分阶数、AR阶数和MA阶数）通常需要一定的调整，我会使用之前确定的参数（p=1, d=1, q=1）进行建模：
import statsmodels.api as sm
model = sm.tsa.ARIMA(endog=endog_train, exog=exog_train, order=(1, 1, 1))
fit_model = model.fit()# disp 参数已经在最新版的 statsmodels 中不再支持了

#进行预测
forecast = fit_model.forecast(steps=len(endog_test), exog=exog_test)#预测未来24天
forecasted_values = forecast

#使用模型在训练集上进行预测
train_forecast = fit_model.predict(start=exog_train.index[0], end=exog_train.index[-1], exog=exog_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error
# 计算训练集的误差
train_mae = mean_absolute_error(endog_train, train_forecast)
train_mse = mean_squared_error(endog_train, train_forecast)
train_rmse = np.sqrt(train_mse)
train_mape = np.mean(np.abs((endog_train - train_forecast) / endog_train)) * 100

# 计算测试集的误差
test_mae = mean_absolute_error(endog_test, forecasted_values)
test_mse = mean_squared_error(endog_test, forecasted_values)
test_rmse = np.sqrt(test_mse)
test_mape = np.mean(np.abs((endog_test - forecasted_values) / endog_test)) * 100

(train_mae, train_mape, train_mse, train_rmse), (test_mae, test_mape, test_mse, test_rmse)
#很抱歉，我们在计算MAPE（平均绝对百分比误差）时遇到了问题，导致了一个无穷大的值。
#这可能是由于数据集中有些值为0，导致在计算MAPE时出现了除以0的情况。

