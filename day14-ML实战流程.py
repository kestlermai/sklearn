# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:16:24 2022

@author: 11146
"""

#ML文章大致流程
#1.数据收集
#纳入、排除标准，技术路线图
#（1）你要纳入多少病人？（越多越好？）
#（2）研究的期限是啥？（XX年至XX年）
#（3）纳入标准：诊断标准是啥？是否有其他限制（年龄呀、治疗方式呀等等）。
#（4）排除标准：为啥要排除，给出科学（能圆的过去）的理由。

#2.数据清洗
#数据清理意味着过滤和“修改”数据（注意，这个修改不是那个修改），使其更易于探索、理解和建模。
#过滤是指去掉不想要或不需要的部分，这样就不需要查看或处理它们。“修改”是指数据的的格式不是我们需要的，需要修正。
#常见的有：缺失值处理、离群值、坏数据、重复数据、不相关的特征、归一化和标准化等等。

import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径
#导入数据类型格式
import numpy as np  #导入numpy包，简写为np
import pandas as pd  #导入pandas包，简写为pd
dataset = pd.read_csv('X disease code.txt')  #注意此时是txt格式了
from sklearn.preprocessing import LabelEncoder
X = dataset.values
labelencoder_X = LabelEncoder()
for i in range(9):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])



#缺失值处理
#当缺失值少于30%时，连续变量可以使用均值或中位数填补。分类变量也可以填补，单算一类即可，或者也可以用众数填补分类变量（就是哪一类多算哪一类）。
#当缺失值大于30%，可以考虑放弃这个特征或者样本了。如果实在很重要，也可以先试试使用上面的填补办法进行填充，看看最终建模效果。

#查找缺失值：
dataset.isnull().any() #统计dataset里每一列是否有缺失值
dataset.isnull().any().sum() #统计dataset里每一列缺失值个数
#不过有时候呢，你的缺失值并不是NA，而是null，所以会识别不出来，得先替换：
dataset = dataset.replace('null',np.NaN)
#还是不出，那就是你的缺失值输入成其他东西了，比如“-”、“/”等等，那就得自己去查找了替换了。

#重复值
#识别重复值
a = dataset.duplicated() #使用所有的列进行检测
b = dataset.duplicated("A") #仅仅考虑A列

#查看有多少重复值
c = dataset.duplicated().sum()
d = dataset.duplicated("A").sum()

#删除重复记录
dataset.drop_duplicates(inplace=True) #inplace=True表示直接在源数据上操作

#异常值处理
#识别；
#1.统计学方法
e = dataset.describe()
#2.箱型图法
import matplotlib.pyplot as plt
dataset['R'].plot(kind='box')
dataset['N'].plot(kind='box')
plt.show()

#异常值的处理方法
#直接将含有异常值的记录删除；
#将异常值视为缺失值，利用缺失值处理的方法进行处理；
#不管它，直接建模。
#是否要删除异常值可根据实际情况考虑。因为一些模型对异常值不很敏感，即使有异常值也不影响模型效果，
#但是一些模型比如逻辑回归对异常值很敏感，如果不进行处理，可能会出现过拟合等非常差的效果。



#3.特征工程
#数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已
#这一步一般是通过差异分析初步筛选有显著性差异的特征，接着结合相关性矩阵，重要性评分再剔除一些特征。

#计数资料和计量资料（连续变量）一般不能同时做相关分析，所以一般都是只做计量资料。
#至于两者（计数资料和计量资料）是否有相关性，硬要做也是可以的，有一方法叫做Kendall's tau-b相关系数。


import numpy as np
import pandas as pd
dataset = pd.read_csv('X disease code.csv')
X = dataset.iloc[:, 1:15] #取自变量
X1 = dataset.iloc[:,6:14] #取自变量中的连续变量
new_X1 = X1.corr() #默认的是Pearson
new_X = X.corr('kendall') #自选Kendall's tau-b

#可以用spss做也可以在python做
DataFrame.corr(method='pearson', min_periods=1) #method：可选值为{‘pearson’, ‘kendall’, ‘spearman’}
#但在python里面只能两两计算P-value，推荐在spss算



#做相关性矩阵-热图
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('cor-heatmap.csv', header = 0, index_col = 0)

import seaborn as sns
plt.subplots(figsize=(9, 9))
sns.heatmap(dataset, annot=True, vmax=1, square=True, cmap="Blues")
plt.show()

#设置格式
import seaborn as sns#可视化库
plt.subplots(figsize=(9, 9))#figsize设置图形的大小
sns.heatmap(dataset, annot=True, square=True, vmin=-0.4 ,vmax=0.4, cmap='GnBu',
            annot_kws={'size':9,'weight':'bold', 'color':'black'},fmt='.1f')#fmt='.1f'保留小数后一位
plt.show()

#输出相关对角矩阵

mask = np.zeros_like(dataset, dtype=np.bool)   #定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型
mask[np.triu_indices_from(mask)]= True      #返回矩阵的上三角，并将其设置为true
plt.subplots(figsize=(9, 9))#figsize设置图形的大小
sns.heatmap(dataset, annot=True, square=True, vmin=-0.4 ,vmax=0.4, cmap='GnBu',mask=mask,
            annot_kws={'size':9,'weight':'bold', 'color':'black'},fmt='.1f')
plt.show()

seaborn.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
                annot=None, fmt=’.2g’, annotkws=None, linewidths=0, linecolor=‘white’, 
                cbar=True, cbarkws=None, cbar_ax=None, square=False, ax=None, 
                xticklabels=True, yticklabels=True, mask=None, **kwargs)
data：输入的相关矩阵数据；
vmin, vmax：用于显示图例中最小值与最大值的值；
cmap：用于设置热图的颜色；
center：可以调整热图的颜色深浅；
annot：设置为True，热图的每个单元上显示具体数值；
fmt：指定单元格中数据的显示格式；
annot_kws：有关单元格中数值标签的其他属性描述，如颜色、大小等；
linewidths：设置每个单元格的边框宽度；
linecolor：设置每个单元格的边框颜色；
cbar：是否用颜色条作为图例，默认为True；
square：是否使热力图的每个单元格为正方形，默认为False
cbar_kws：有关颜色条的其他属性描述；
xticklabels,yticklabels：设置x轴和y轴的刻度标签，如果为True，则分别以数据框的变量名和行名称作为刻度标签；
mask：用于突出显示某些数据；
ax：用于指定子图的位置。

用R画热图更方便

#重要性评分
#lasso回归
#https://blog.csdn.net/weixin_39707597/article/details/113085862 （用R）
#https://blog.csdn.net/sinat_41858359/article/details/124765520 （用python）

#2.RF
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
dataset = pd.read_csv('X disease code.csv')
X = dataset.iloc[:, 1:14].values
Y = dataset.iloc[:, 0].values
sc = StandardScaler()
X_train = sc.fit_transform(X)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y)

#输出重要指数
importance = classifier.feature_importances_#输出重要性指标
imp_result = np.argsort(importance)[::-1][:10] #只取前十
index = np.argsort(importance)[::-1] #取全部
feat_labels = dataset.columns[1:]#截取特征名称，特征名称和重要性指标的数值是一一对相应的
for i in range(len(imp_result)):
    print("%2d. %-*s %f" % (i + 1, 30, feat_labels[imp_result[i]], importance[imp_result[i]]))#从高到低输出
for i in range(len(index)):
    print("%2d. %-*s %f" % (i + 1, 30, feat_labels[index[i]], importance[index[i]]))
# %d就是输出了整形数
# %2d是将数字按宽度为2，采用右对齐方式输出，若数据位数不到2位，则左边补空格。
# %02d，和%2d差不多，只不过左边补0
# %-2d将数字按宽度为2，采用左对齐方式输出，若数据位数不到2位，则右边补空格
# %.2d 输出整形时最少输出2位，如不够前面以0占位。如输出2时变成02，200时只输出200；输出浮点型时（%.2f）小数点后强制2位输出

# %-*s 代表输入一个字符串，-号代表左对齐、后补空白，*号代表对齐宽度由输入时确定
# %*s 代表输入一个字符串，右对齐、前补空白，*号代表对齐宽度由输入时确定

# %f 输出浮点型
#%格式化输出，如a='test',print('it is a %s'%a)=it is a test


import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')#更改matplot默认字体

feat_labels = [feat_labels[i] for i in imp_result]
plt.title('你的数据集特征的重要程度')
plt.bar(range(len(imp_result)), importance[imp_result], color='lightblue', align='center')#bar为每个元素绘制一个纵向条形
#barh函数得到横向条形图
#align=center居中对齐
plt.xticks(range(len(imp_result)), feat_labels, rotation=90)#xticks为坐标轴刻度
plt.xlim([-1, len(imp_result)])#x轴的范围
plt.tight_layout()#自动调整子图参数的函数
plt.show()

#如何选择？
#可以看到，不同模型得出来的结果各有特色（差距有点大），那么怎么选呢？其实没有固定策略，提供几种供参考：
#以一种模型为主，写文章就写其中一种，其他模型结果当做没看见；
#也可以综合考虑，少数服从多数，往往比较难；
#实在不行，每个模型输出的特征重要性都考虑，也就是：分别根据这些特征的重要性结合相关矩阵进行筛选，然后分别建模，得出N个模型，再比较；
#不同的数据会出现不同的情况，非常考验随机应变能力。
#不过有一个原则是不变的：每一步都有对应的、科学严谨（审稿人质疑的时候能够理直气壮地回怼）的筛选标准。
#此外，也不是一蹴而就的，很多时候到后面建模发现问题，还是要回来重新调整特征工程的，来来回回好几次。

#4.构建模型
#这一步就是建模了，之前提到的模型，可以都走一遍。不过一般也不用全上，
#我个人喜欢用逻辑回归作为对照模型，然后上K-NN、随机森林、Xgboost、LightGBM、Catboost和支持向量机。

#5.模型性能评价
#（1）训练集、测试集的ROC和AUC
#（2）训练集、测试集的混淆矩阵
#（3）训练集、测试集的PR曲线
#（4）训练集、测试集的性能指标
#（5）SHAP-shapley value沙普利值

#6.误判数据分析
#对于那些被误判的样本弄出来，分析其特征，卡方检验、小提琴图+t检验