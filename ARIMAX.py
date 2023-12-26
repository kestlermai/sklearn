# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:34:39 2023

@author: maihuanzhuo
"""

####ARIMAX模型
#ARIMAX模型是指带回归项的ARIMA模型，又称扩展的ARIMA模型。
#实际情况中很多序列的变化规律会受到其它序列的影响，针对这种情况需要建立多元时间序列的ARIMAX模型。

#假设响应序列y[t]和自变量序列x[1t],x[2t], …, x[kt]均平稳，且响应序列和输入序列之间具有线性相关关系。
#假设第i个自变量序列x[it]对响应变量序列y[t]的影响要延迟l[i]期发挥作用，有效作用期长为n[i]期。
#ARIMAX回归模型引入了自回归系数多项式和移动平均多项式结构，可称为传递函数模型。
#该部分并没有考虑到时间序列的滞后性问题，即今天的某个变量因素在未来1,2,3,..,n天内会影响结局变化。
#所以可以计算结局跟变量的滞后值的相关性，确定在哪个滞后值lag下与结局之间的相关性最强。

#示例：----------------https://www.heywhale.com/mw/project/60ed9dd93aeb9c0017bb836f/content
#假设加入外部变量标普500对黄金价格预测有一定影响

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/时间序列建模/ARIMAX') ##修改路径
#读取金价数据
import pandas as pd 
import numpy as np

gold = pd.read_csv('gold-price-last-ten-years.csv', sep=',', header=8)
gold.head()

#将时间列转换为时间格式
gold['date']=pd.to_datetime(gold['date'])

#利用focus布尔因素（其实就是判断true or false）筛选出日期大于'2017-08-21'且小于等于'2020-08-21'的行
focus = (gold['date'] > '2017-08-21') & (gold['date'] <= '2020-08-21')
#存储到maindf中
maindf = gold.loc[focus]
maindf.head()

#更改列名
maindf.columns = ['Date', 'GLD(USD)']
maindf.head()

#检查缺失值
#定义一个缺失值函数
def show_na(df):
    # 创建一个新的DataFrame na_df，其中包含了每列的缺失值数量和缺失值的百分比
    na_df = pd.concat([df.isnull().sum(), 100 * df.isnull().mean()], axis=1)
    # # 为na_df的列命名为'count'和'%'，以分别表示缺失值数量和缺失值百分比
    na_df.columns=['count', '%']
    # 按缺失值数量降序排序na_df，并返回结果
    na_df.sort_values(by='count', ascending = False)
    return na_df
show_na(maindf)#没有缺失值

#读取SP500数据
SP500=pd.read_csv('SP500 2007-2020.csv', sep=',', header=7)
SP500['date'] = pd.to_datetime(SP500['date'])
SP500.columns = ['Date', 'SPX(USD)']
#根据上面金价数据的data列对SP500的进行匹配，指定为左连接
maindf = maindf.merge(SP500, how='left', on='Date')
show_na(maindf)#检查缺失值

#同样读取巴里克黄金公司的金价和白银价格
barrick = pd.read_csv('Barrick Gold Corp 1985-2020.csv', sep=',', header=9)
barrick['date'] = pd.to_datetime(barrick['date'])
barrick = barrick[['date', 'close']]
barrick.columns = ['Date','BARR(USD)']
maindf = maindf.merge(barrick, how='left', on='Date')
maindf.head()

#白银价格
silver = pd.read_csv('silver history.csv', sep=',', header=7)
silver['date']=pd.to_datetime(silver['date'])
silver.columns = ['Date', 'SLV(USD)']
maindf = maindf.merge(silver, how='left', on='Date')
maindf.head()

#展示三个外部因素的缺失值
show_na(maindf)
#检查日期是否有重复值
maindf['Date'].value_counts().head()

#Set index to 'Date'  for graphing and visualization 将索引设置为“日期”以进行绘图和可视化
maindf= maindf.set_index('Date')
maindf.head()

maindf.describe()

#
#通过ffill（pandas中forward fill向前填充，即通过填充前面一个值进行填补）进行缺失值填补
maindf = maindf.fillna(axis=0, method='ffill')
maindf.describe()#变化不会很大
#检查是否还有缺失值
maindf.isnull().sum()

import matplotlib.pyplot as plt 
from matplotlib.pylab import rcParams#用于配置Matplotlib图形属性的对象，它允许你自定义图形的各种参数，如图形大小、线条颜色、字体大小
#时间序列图可视化
plt.figure(figsize=(20,15))
plt.subplot(411)
plt.plot(maindf['GLD(USD)'], label='GLD', color='gold')
plt.legend(loc='best', fontsize='large')
plt.ylabel('Value(USD)')
plt.subplot(412)
plt.plot(maindf['SPX(USD)'], label='S&P', color='blue')
plt.legend(loc='best', fontsize='large')
plt.ylabel('Value(USD)')
plt.subplot(413)
plt.plot(maindf['BARR(USD)'], label='BARR', color='green')
plt.legend(loc='best', fontsize='large')
plt.ylabel('Value(USD)')
plt.subplot(414)
plt.plot(maindf['SLV(USD)'], label='SLV', color='silver')
plt.legend(loc='best', fontsize='large')
plt.xlabel('Date')
plt.ylabel('Value(USD)')
#在2020年第一次封锁前后，我们可以清楚地看到，金价随着SP500的减少而增加。与此同时，巴里克黄金公司（BARR）和白银价格似乎也倾向于金价。

import seaborn as sns 
#Using jointplot to visualize relation of GLD with other variables
sns.jointplot(x=maindf['SPX(USD)'], y=maindf['GLD(USD)'], color='gold', kind="reg",height=6, ratio=8, marginal_ticks=True)
sns.jointplot(x=maindf['SPX(USD)'], y=maindf['BARR(USD)'], color='green', kind="reg",height=6, ratio=8, marginal_ticks=True)
sns.jointplot(x=maindf['SPX(USD)'], y=maindf['SLV(USD)'], color='silver', kind="reg",height=6, ratio=8, marginal_ticks=True)

#相关性热图
corr = maindf.corr()
plt.figure(figsize = (8,6))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True, fmt='.2f', linewidths=.30)
plt.title('Correlation of MainDF Features', y =1.05,  size=15)
pos, textvals = plt.yticks()
plt.yticks(pos,('GLD(USD)','SPX(USD)','BARR(USD)','SLV(USD)'), 
    rotation=0, fontsize="10", va="center")
#看起来黄金（GLD）的价值与SPX的价值显著相关。此外，白银与巴里克黄金公司的关联是有道理的，因为他们可能也在交易白银。
print(corr['GLD(USD)'].sort_values(ascending =False), '\n')

##分割数据为训练集和测试集
#数据中一共有759个时间点，那么选择80%为训练集，20%为测试集，即选取前面607个时间点为训练集，152个时间点为测试集
n_obs = 152
X_train, X_test = maindf[0:-n_obs], maindf[-n_obs:]
print(X_train.shape, X_test.shape)

X_train_transformed = X_train.diff().dropna()
X_train_transformed.head()
X_train_transformed.describe()

#时间平稳性检验ADF
from statsmodels.tsa.stattools import adfuller
def augmented_dickey_fuller_statistics(time_series):
    result = adfuller(time_series.values)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
        
print('Augmented Dickey-Fuller Test: Gold Price Time Series')
augmented_dickey_fuller_statistics(X_train_transformed['GLD(USD)'])#p-value: 0.000061
print('Augmented Dickey-Fuller Test: SPX Price Time Series')
augmented_dickey_fuller_statistics(X_train_transformed['SPX(USD)'])#p-value: 0.000000
print('Augmented Dickey-Fuller Test: BARR Price Time Series')
augmented_dickey_fuller_statistics(X_train_transformed['BARR(USD)'])#p-value: 0.000000
print('Augmented Dickey-Fuller Test: Silver Price Time Series')
augmented_dickey_fuller_statistics(X_train_transformed['SLV(USD)'])#p-value: 0.000000

#stationary
fig, axes = plt.subplots(nrows=4, ncols=1, dpi=120, figsize=(20,20))
for i, ax in enumerate(axes.flatten()):
    d = X_train_transformed[X_train_transformed.columns[i]]
    ax.plot(d, color='red', linewidth=1)
    ax.set_title(X_train_transformed.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_alpha(0)
    ax.tick_params(labelsize=8)
plt.show();

# now that they are stationary we can test for Granger Causality!
# 格兰杰因果关系用于检验一组时间序列是否为另一组时间序列的原因,探讨两者之间的因果关系。
# 假如A是B的格兰杰原因，则说明A的变化是引起B变化的原因之一。

from statsmodels.tsa.stattools import grangercausalitytests
maxlag = 12
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

grangers_causality_matrix(X_train_transformed, variables = X_train_transformed.columns)

#             GLD(USD)_x  SPX(USD)_x  BARR(USD)_x  SLV(USD)_x
#GLD(USD)_y       1.0000      0.0388       0.0000      0.0000
#SPX(USD)_y       0.0642      1.0000       0.0077      0.0043
#BARR(USD)_y      0.1356      0.0127       1.0000      0.0024
#SLV(USD)_y       0.5869      0.1145       0.0085      1.0000

#可以看到GLD(USD)_x与SPX(USD)_y，BARR(USD)_y，SLV(USD)_y 没有因果关系（p>0.05）
#而SPX(USD)_x，BARR(USD)_x，SLV(USD)_x与GLD(USD)_y存在因果关系（p<0.05）

#ARIMAX建模
date = maindf.index
X = maindf['GLD(USD)']
size = int(len(X)*0.8)
train, test = X[0:size], X[size:len(X)]
date_test = date[size:]

#定义计算RMSE平均平方误差的根
def RMSEfromResid(X):
    summ = 0
    for i in X:
        summ+=i**2
    return((summ/len(X))**0.5)

import statsmodels.api as sm
def evaluate_arima_model(X, model_order):
    model_arima = sm.tsa.ARIMA(X, order=model_order).fit()#disp=0这个参数被移除了？
    AIC = model_arima.aic
    BIC = model_arima.bic
    LLF = model_arima.llf
    RMSE = RMSEfromResid(model_arima.resid)
    return([AIC, BIC, LLF, RMSE])

# 设置参数范围
p_values = [0,1,2,3]
d_values = [1]
q_values = [0,1,2]
data = list()
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            try:
                [AIC, BIC, LLF, RMSE] = evaluate_arima_model(train, order)
                data.append([order,AIC, BIC, LLF, RMSE])
            except:
                continue
ARIMA_Models = pd.DataFrame(data,columns=['ARIMA', 'AIC', 'BIC', 'Maximum Log-Likelihood', 'RMSE'], dtype=float)
evaluate_arima_model(X, order)
ARIMA_Models.sort_values(by=['RMSE'])#原作者中RMSE值为9.114403，不过其他三个值基本一致
#按照AIC值从低到高排序
ARIMA_Models.sort_values(by=['AIC'])

#ARIMA Prediction
history = [x for x in train]
predictions = list()
data=list()
#len_test = len(test)
len_test= len(test)
for t in range(len_test):
    model_arima = sm.tsa.ARIMA(endog = history, order=(1, 1, 1)).fit()#作者选取（2,1,1），一般选AIC和BIC更小，Log-Likelihood更大，RMSE更小
    #(2, 1, 1)	4408.594196	4430.628596	-2199.297098	9.117948
    output = model_arima.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    data.append([date_test[t], obs, yhat])
    
from sklearn.metrics import mean_squared_error  
RMSE = (mean_squared_error(test, predictions))**0.5
arima_results = pd.DataFrame(data,columns=['Period','Actual Price', 'Predicted Price'],dtype=float)
print('Test RMSE: %.3f' % RMSE)
#Test RMSE: 22.856（2,1,1），Test RMSE: 22.668（1,1,1）跟作者结果（Test RMSE: 22.664，（2,1,1））基本一致

print(model_arima.summary())#这里不知道为什么是SARIMAX Results 

def mape(y_true, y_pred):
    return np.sum(((y_pred - y_true) / y_pred)) * 100

print('The Mean Absolute Percentage Error is: %.3f' % mape(np.array(test[1:]), predictions[:-1]),'%.')
#The Mean Absolute Percentage Error is: -44.067 %.，作者：The Mean Absolute Percentage Error is: -18298.529 %.

plt.rcParams['figure.figsize'] = (20,10)
plt.plot(date_test, test, color='Blue', label='ACTUAL', marker='x')
plt.plot(date_test, predictions, color='green', label='PREDICTED', marker='x')
plt.legend(loc='upper right')
plt.show()
arimax_pred = predictions
arimax_RMSE = RMSE
arimax_RMSE

# ARIMAX Train-Test-Split:
maindf['diffGLD'] = maindf['GLD(USD)'].diff()#进行差分处理
maindf['diffSPX'] = maindf['SPX(USD)'].diff()#进行差分处理
maindf['SPX_lag'] = maindf['diffSPX'].shift()#存储了标普500指数差分的滞后一期值（lag=1）
maindf.dropna(inplace=True)#删除了包含NaN值的行，通常是因为差分操作导致的首行数据丢失
GLD_end = maindf['GLD(USD)']
SPX_ex = maindf['SPX_lag']
m = len(GLD_end)
size = int(len(GLD_end)*0.8)
train, test = GLD_end[0:size], GLD_end[size:m]
ex_train, ex_test = SPX_ex[0:size], SPX_ex[size:m]
date_test = date[size:]

def evaluate_arimax_model(y, X, model_order):
    model_arimax = sm.tsa.ARIMA(endog = y, exog=X, order=model_order).fit()
    AIC = model_arimax.aic
    BIC = model_arimax.bic
    LLF = model_arimax.llf
    RMSE = RMSEfromResid(model_arimax.resid)
    return([AIC, BIC, LLF, RMSE])

p_values = [0,1,2,3]
d_values = [1]
q_values = [0,1,2]
data = list()
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            try:
                [AIC, BIC, LLF, RMSE] = evaluate_arimax_model(train, ex_train, order)
                data.append([order,AIC, BIC, LLF, RMSE])
            except:
                continue

ARIMAX_Models = pd.DataFrame(data,columns=['ARIMAX', 'AIC', 'BIC', 'Maximum Log-Likelihood', 'RMSE'], dtype=float)

evaluate_arimax_model(train, ex_train, order)
ARIMAX_Models.sort_values(by=['RMSE'])
ARIMAX_Models.sort_values(by=['AIC'])


history = [x for x in train]
his_u = ex_train
predictions = list()
data = list()
test_index = list()
for t in range(len(ex_test)):
    model_arimax = sm.tsa.ARIMA(endog = history, exog=his_u, order=(1,1,1)).fit()#exog用来传递外生变量
    #使用已拟合的ARIMAX模型，对未来一个时间步（steps=1）进行预测，同时提供测试集对应时间步的外生变量值
    output = model_arimax.forecast(steps = 1, exog = ex_test.iloc[[t]])
    #yhat = output[0]#报错，删掉[0]又可以跑了
    yhat = output
    predictions.append(yhat)
    history.append(test[t])
    test_index.append(t)
    his_u = ex_train.append(ex_test.iloc[test_index])
    data.append([date_test[t], test[t], yhat])
    
output_list = []  # 创建一个空列表来存储输出
for t in range(len(ex_test)):
    model_arimax = sm.tsa.ARIMA(endog=history, exog=his_u, order=(1, 1, 1)).fit()
    output = model_arimax.forecast(steps=1, exog=ex_test.iloc[[t]])
    print(output)  # 打印output内容
    output_list.append(output[0])  # 将output添加到output_list中

RMSE = (mean_squared_error(test, predictions))**0.5
arima_results = pd.DataFrame(data,columns=['Period','Actual Price', 'Predicted Price'],dtype=float)
print('Test RMSE: %.3f' % RMSE)
#Test RMSE: 22.936
print('The Mean Absolute Percentage Error is: %.3f' % mape(np.array(test), predictions),'%.')
#The Mean Absolute Percentage Error is: -16039.317 %.

plt.rcParams['figure.figsize'] = (20,10)
plt.plot(date_test[2:], test, color='Blue', label='ACTUAL', marker='x')
plt.plot(date_test[2:], predictions, color='green', label='PREDICTED', marker='x')
plt.legend(loc='upper right')
plt.show()
arimax_pred = predictions
arimax_RMSE = RMSE
arimax_RMSE
#22.936248175925876

print(model_arimax.summary())

#作者结论：外部变量标普500对黄金价格的预测有一定影响，但是可能没有那么大
#在我这结果只有丝毫变化，在2020-03到2020-04底部影响预测往下趋势