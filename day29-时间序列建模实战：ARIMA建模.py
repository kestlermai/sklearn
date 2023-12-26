# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:04:59 2023

@author: maihuanzhuo
"""

#时间序列建模实战：ARIMA建模
#ARIMA（自回归积分滑动平均）模型
#SARIMA（季节性自回归积分滑动平均）模型
import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模') ##修改路径

#读取数据
import pandas as pd
data = pd.read_csv('data.csv')
data.head()

import matplotlib.pyplot as plt
# 绘制时间序列图
plt.figure(figsize=(15, 6))
plt.plot(data['incidence'], label='月发病率')
plt.title('肾综合症出血热月发病率时间序列图')
plt.xlabel('月份')
plt.ylabel('发病率')
plt.legend()
plt.grid(True)
plt.show()

#接着我们需要进行平稳性检验（ADF（Augmented Dickey-Fuller）检验来判断序列的平稳性），只有平稳的时间序列才能用于建立SARIMA/ARIMA
from statsmodels.tsa.stattools import adfuller
# 进行ADF检验
result = adfuller(data['incidence'])
adf_statistic, p_value, usedlag, nobs, critical_values, icbest = result
# 输出ADF检验结果
adf_result = {
    "ADF统计量": adf_statistic,
    "p值": p_value,
    "滞后阶数": usedlag,
    "观测值数量": nobs,
    "临界值": critical_values,
    "最大信息准则": icbest
}
adf_result
#{'ADF统计量': -4.140878665088878,
# 'p值': 0.0008283345418149487,
# '滞后阶数': 11,
# '观测值数量': 96,
# '临界值': {'1%': -3.5003788874873405,
#  '5%': -2.8921519665075235,
#  '10%': -2.5830997960069446},
# '最大信息准则': -571.3274452014111}
#由于p值小于0.05，并且ADF统计量小于各个临界值，我们可以拒绝原假设，即序列存在单位根。因此，我们可以认为该时间序列是平稳的。
#如果不通过则采用差分方法，即将原始时间序列的差分序列作为新的时间序列进行分析。差分序列可以消除时间序列的趋势和季节性，使其更具有平稳性。

#我们将进行自相关（ACF）和偏自相关（PACF）图的绘制，以便选择合适的SARIMA模型参数
#其实这里还需要做一次白噪声检验 (Ljung-Box检验)，若通过 (P<0.05) 则说明数据具有分析价值
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox_result = acorr_ljungbox(data['incidence'], lags=40, boxpierce=True, return_df=True)
acorr_ljungbox_result

#自相关（ACF）和偏自相关（PACF）图的绘制
#截尾跟拖尾的区别，截尾即当阶数为 1 的时候，系数值还是很大， 0.914， 二阶长的时候突然就变成了 0.050，后面的值都很小，认为是趋于 0
#而拖尾就是拖尾就是有一个衰减的趋势，但是不都为 0，
#自相关图：观察有无超过两倍标准差的范围，超过了就是拖尾没超过就是截尾
#初步确定SARIMA/ARIMA模型的非季节性参数p和q
#（1）如果自相关是拖尾，偏相关截尾，则用 AR 算法；
#（2）如果自相关截尾，偏相关拖尾，则用 MA 算法；
#（3）如果自相关和偏相关都是拖尾，则用 ARMA 算法， ARIMA 是 ARMA 算法的扩展版，用法类似 。

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf#该时间序列是有周期性的，而且周期为12
# 绘制自相关函数（ACF）图
plt.figure(figsize=(12, 6))
plot_acf(data['incidence'], lags=40, title='自相关函数（ACF）图')
plt.show()

# 绘制偏自相关函数（PACF）图
plt.figure(figsize=(12, 6))
plot_pacf(data['incidence'], lags=40, title='偏自相关函数（PACF）图')
plt.show()

## 导入SARIMA模型
from statsmodels.tsa.statespace.sarimax import SARIMAX
# 分割数据为训练和测试集
train_data = data['incidence'][:96]  # 2004年1月至2011年12月的数据用于训练
test_data = data['incidence'][96:]   # 2012年1月至12月的数据用于测试

#一般来说，疾病发病率的ARIMA模型p，q,P,Q值一般都在0,1,2里面选，而差分阶数一般为1或者0。
# 设置参数范围
p_values = range(0, 2)
d_values = range(0, 1)
q_values = range(0, 2)
P_values = range(0, 2)
D_values = range(0, 1)
Q_values = range(0, 2)
s_value = 12#seasonal_order

#基于AIC值、SBC值和模型参数的统计学意义来选择最优模型，一般根据AIC或BIC最小原则选择较优模型
# 用于存储最优模型的信息
best_model_info = {
    "AIC": float('inf'),
    "SBC": float('inf'),
    "R_squared": float('-inf'),
    "order": None,
    "seasonal_order": None,
    "model": None
}
# 遍历参数，寻找最优模型
for p in p_values:
    for d in d_values:
        for q in q_values:
            for P in P_values:
                for D in D_values:
                    for Q in Q_values:
                        try:
                            # 构建SARIMA模型
                            model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s_value), enforce_invertibility=False)
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
                                    best_model_info["seasonal_order"] = (P, D, Q, s_value)
                                    best_model_info["model"] = model_fit
                        except:
                            continue

# 打印最优模型信息
best_model_info["order"], best_model_info["seasonal_order"], best_model_info["AIC"], best_model_info["SBC"]
#((1, 0, 0), (1, 0, 1, 12), -554.4732257947044, -544.215833028833)

# 获取最优模型各个参数的p值
p_values_of_best_model = best_model_info["model"].pvalues

# 打印各个参数的p值
print("最优模型各个参数的p值：")
for param, p_value in p_values_of_best_model.items():
    print(f"{param}: {p_value}") 
#最优模型各个参数的p值：
#ar.L1: 4.4040863221401144e-11
#ar.S.L12: 0.0
#ma.S.L12: 5.457174769724076e-09
#sigma2: 3.2885292904613513e-12

##拟合模型诊断：检验拟合模型的准确性，提供模型改进方向
#需要（1）做残差正态性检验QQ图，Shapiro-Wilk正态性检验
#（2）检验残差自相关函数值，残差白噪声检验

model = SARIMAX(data['incidence'], order=(1,0,0), seasonal_order=(1,0,1,12))
result = model.fit()
print(result.summary())

from statsmodels.graphics.api import qqplot
resid = result.resid#残差
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

# Forecast
forecast = model_fit.forecast(len(test_data))
import numpy as np
forecast = np.exp(forecast)
# Plot对比预测值与验证集
plt.figure(figsize=(15,5))
plt.plot(np.exp(train_data), label='Training data')
plt.plot(np.exp(test_data), label='Validation data')
plt.plot(forecast,label='SARIMA')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.title('Prediction by SARIMA')
plt.legend()
plt.show()

#再建一个ARIMA模型用作两个模型对比：
import statsmodels.api as sm
model.ARIMA = sm.tsa.ARIMA(data['incidence'], order=(1,0,0), seasonal_order=(1,0,1,12))
result.ARIMA = model.ARIMA.fit()

# Forecast on validation set
forecast.ARIMA = result.ARIMA.predict(len(train_data),#指定开始预测的位置，即训练集的长度
                                      len(data['incidence'])-1,#指定结束预测的位置,即时间序列的总长度减去1。它告诉模型预测直到时间序列的最后一个观察值。
                                      typ='levels',#指定要在原始数据水平上获取预测，而不是在差分形式上获取
                                      dynamic=False)#预测是使用实际值（而不是之前预测的值）作为输入进行的
forecast.ARIMA = np.exp(forecast.ARIMA)
#可视化对比
plt.figure(figsize=(15,5))
plt.plot(np.exp(train_data), label='Training data')
plt.plot(np.exp(test_data), label='Validation data')
plt.plot(forecast.ARIMA,label='ARIMA')
plt.xlabel('Date')
plt.ylabel('Passengers')
plt.title('Prediction by ARIMA')
plt.legend()
plt.show()

#用RMSE、R2和MAPE来评估模型拟合结果
# define MSE function
def mse(prediction, actual):
    return ((prediction-actual)**2).mean()

# define R2 function
def r2(prediction, actual):
    return 1-mse(prediction, actual)/np.var(actual)

# define MAPE function
def mape(prediction, actual):
    return ((((actual - prediction)/actual) * 100).abs()).mean()
#RMSE的数值越小越好，表示模型的预测越准确
RMSE_SARIMA = mse(forecast, np.exp(test_data))**0.5#**0.5代表开方运算即平方根
RMSE_ARIMA = mse(forecast.ARIMA, np.exp(test_data))**0.5
#R²越接近1，表示模型越能够解释数据的变异性，拟合效果越好
r2_SARIMA = r2(forecast, np.exp(test_data))
r2_ARIMA = r2(forecast.ARIMA, np.exp(test_data))
#MAPE的数值越小越好，表示模型的平均百分比误差越低
MAPE_SARIMA = mape(forecast, np.exp(test_data))
MAPE_ARIMA = mape(forecast.ARIMA, np.exp(test_data))

print('RMSE of SARIMA: {:.3f}'.format(RMSE_SARIMA))
print('RMSE of ARIMA: {:.3f}'.format(RMSE_ARIMA))

print('R2 of SARIMA: {:.3f}'.format(r2_SARIMA))
print('R2 of ARIMA: {:.3f}'.format(r2_ARIMA))

print('MAPE of SARIMA: {:.3f}%'.format(MAPE_SARIMA))
print('MAPE of ARIMA: {:.3f}%'.format(MAPE_ARIMA))

#最后拟合和预测数据
# 使用最优模型进行全样本内预测（2004年1月至2011年12月）
in_sample_forecast = best_model_info["model"].get_prediction(dynamic=False)
in_sample_mean = in_sample_forecast.predicted_mean
in_sample_conf_int = in_sample_forecast.conf_int()
# 使用最优模型进行样本外预测（2012年1月至12月）
out_sample_forecast = best_model_info["model"].get_forecast(steps=12)
out_sample_mean = out_sample_forecast.predicted_mean
out_sample_conf_int = out_sample_forecast.conf_int()

# 保存in_sample_conf_int的第二列为CSV文件
in_sample_conf_int.iloc[:, 1].to_csv('in_sample_conf_int_second_column.csv', header=['incidence'])
# 保存out_sample_conf_int的第二列为CSV文件
out_sample_conf_int.iloc[:, 1].to_csv('out_sample_conf_int_second_column.csv', header=['incidence'])
#拟合和预测数据输出为CSV文件，自行取用作图即可

#观察模型拟合精度
#①ME：平均误差；
#②RMSE：均方根误差；
#③MAE：平均绝对误差，范围[0,+∞)，当预测值与真实值完全吻合时等于0，即完美模型；
#④MPE：平均百分比误差；
#⑤MAPE：平均绝对百分误差，范围[0,+∞)，MAPE 为0%表示完美模型，MAPE大于 100 %则表示劣质模型；
#⑥MASE：平均绝对标准化误差。