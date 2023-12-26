# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 21:31:20 2022

@author: 11146
"""

#logistics回归建模-调参

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test') ##修改路径
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('X disease code.csv')
X = dataset.iloc[:, 1:14].values
Y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 666)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs')#先试试'lbfgs'
##solver优化算法的参数，包括newton-cg,lbfgs,liblinear,sag,saga,对损失的优化的方法
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_trainpred = classifier.predict(X_train)
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

#总体来看，大同小异，硬要比较，solver='lbfgs'稍微好一些。不过呢，有一丢丢存在过拟合了：训练集准确率稍微大于验证集。
#因此，稍微调整一下惩罚项penalty以及C：
#不过呢，当选择solver='lbfgs'时，penalty只能选则‘l2’，所以只能把C调小试一试效果：
classifier = LogisticRegression(solver='lbfgs', C=0.1)

#random_state = 设置为其他数字看看。
#也就等1314的时候，稍微好一点：均上81%。
#会不会存在比1314还好的数据构成？要怎么寻找呢？

#调参
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
#predict_proba函数可以获得对每种可能结果的概率,大于0.5为1类，否则为0类

y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

#对训练集可视化
classes = list(set(y_test))#set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
classes.sort()#sort函数对列表元素进行排序
plt.imshow(cm_train, cmap=plt.cm.Blues)#imshow热图，cmap颜色图谱
indices = range(len(cm_train))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()#添加颜色条形
plt.xlabel('predict')#x轴标签
plt.ylabel('real')#y轴标签
for first_index in range(len(cm_train)):
    for second_index in range(len(cm_train[first_index])):
        plt.text(first_index, second_index, cm_train[first_index][second_index])
plt.show()

#对测试集可视化
classes = list(set(y_test))#set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
classes.sort()#sort函数对列表元素进行排序
plt.imshow(cm_test, cmap=plt.cm.Blues)#imshow热图，cmap颜色图谱
indices = range(len(cm_test))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()#添加颜色条形
plt.xlabel('predict')#x轴标签
plt.ylabel('real')#y轴标签
for first_index in range(len(cm_test)):
    for second_index in range(len(cm_test[first_index])):
        plt.text(first_index, second_index, cm_test[first_index][second_index])
plt.show()

#https://www.zhihu.com/question/30643044
#灵敏度、特异度等指标
#sensitive=真阳/(真阳+假阴)=d/d+c
#specificity=真阴/(真阴+假阳)=a/a+b
#acc精确率=a+b/a+b+c+d
#error_rate错误率=1-acc

#F1_Score：F1分数（F1-Score），又称为平衡 F分数（Balanced Score），它被定义为正确率和召回率的调和平均数。
#在 β=1 的情况，F1-Score的值是从0到1的，1是最好，0是最差。

#MCC（Matthews correlation coefficient），即马修斯相关系数，它综合地考量了混淆矩阵中的四个基础评价指标,是二分类问题的最佳度量指标
#MCC=d*a-b*c/(math.sqrt((d+b)*(d+c)*(a+b)*(a+c)))

import math
from sklearn.metrics import confusion_matrix,roc_auc_score,auc,roc_curve
cm = confusion_matrix(y_test, y_pred)   
cm_train = confusion_matrix(y_train, y_trainpred)

#测试集的参数
a = cm[0,0]
b = cm[0,1]
c = cm[1,0]
d = cm[1,1]
acc = (a+d)/(a+b+c+d)
error_rate = 1 - acc
sen = d/(d+c)
sep = a/(a+b)
precision = d/(b+d)
F1 = (2*precision*sen)/(precision+sen)
MCC = (d*a-b*c) / (math.sqrt((d+b)*(d+c)*(a+b)*(a+c)))
auc_test = roc_auc_score(y_test, y_testprba)
#auc是实际值与预测概率比较

#训练集的参数
a_train = cm_train[0,0]
b_train = cm_train[0,1]
c_train = cm_train[1,0]
d_train = cm_train[1,1]
acc_train = (a_train+d_train)/(a_train+b_train+c_train+d_train)
error_rate_train = 1 - acc_train
sen_train = d_train/(d_train+c_train)
sep_train = a_train/(a_train+b_train)
precision_train = d_train/(b_train+d_train)
F1_train = (2*precision_train*sen_train)/(precision_train+sen_train)
MCC_train = (d_train*a_train-b_train*c_train) / (math.sqrt((d_train+b_train)*(d_train+c_train)*(a_train+b_train)*(a_train+c_train)))
auc_train = roc_auc_score(y_train, y_trainprba)


#绘画训练集ROC曲线
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_trainprba, pos_label=1, drop_intermediate=False) 
#fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
#该函数的传入参数为目标特征的真实值y_test和模型的预测值y_test_predprob。需要为pos_label赋值，指明正样本的值。
#drop_intermediate默认T，是否删除一些不会出现在ROC曲线上的次优阈值，有助于画ROC曲线
plt.plot([0, 1], [0, 1], '--', color='navy')    
plt.plot(fpr_train, tpr_train, 'k--',label='Mean ROC (area = {0:.4f})'.format(auc_train), lw=2,color='darkorange')
#lw=linewidth线宽
plt.xlim([-0.01, 1.01]) #边界范围
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Please replace your title')
plt.legend(loc="lower right")#图例位置，lower right=loc=4，图例放在图的右下角
plt.show()

#绘画测试集ROC曲线
fpr_train, tpr_train, thresholds_train = roc_curve(y_test, y_testprba, pos_label=1, drop_intermediate=False)
plt.plot([0, 1], [0, 1], '--', color='navy')    
plt.plot(fpr_train, tpr_train, 'k--',label='Mean ROC (area = {0:.4f})'.format(auc_test), lw=2,color='darkorange')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Please replace your title')
plt.legend(loc="lower right")
plt.show()

#PR曲线是敏感的，随着正负样本比例的变化，PR会发生强烈的变化。
#而ROC曲线是不敏感的，其曲线能够基本保持不变。
#“在negative instances的数量远远大于positive instances的数据集里，PR更能有效衡量分类器的好坏。”

#绘画测试集PR曲线precision-recall-curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
precision_1, recall_1, thresholds = precision_recall_curve(y_test, y_testprba)
plt.step(recall_1, precision_1, color='darkorange', alpha=0.2,where='post')
#step函数用于绘制阶梯图,where设置阶梯所在的位置，取值范围'pre','post','mid'
plt.fill_between(recall_1, precision_1, step='post', alpha=0.2,color='darkorange')
plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall_1,precision_1)
plt.show()
AP_test = average_precision_score(y_test, y_testprba, average='macro', pos_label=1, sample_weight=None)
#平均精度average precision score
#'micro'：通过将标签指标矩阵的每个元素视为标签来全局计算指标。
#'macro'：计算每个标签的指标，并找到它们的未加权平均值。这没有考虑标签不平衡。
#'weighted'：计算每个标签的指标，并找到它们的平均值，按支持度加权(每个标签的真实实例数)。
#'samples'：计算每个实例的指标，并找到它们的平均值。
#sample_weight=none,样本权重默认none

#精度-召回曲线绘制了不同概率阈值的精度和召回率p。对于p=0，所有事物都归为1，因此召回率将为100％，
#精度将为测试数据集中1的比例。对于p=1，任何东西都不归类为1，因此召回率将为0％，精度将为0。
#对于p=0.5，这就是precision_score告诉您的信息，但是，您可能不希望在最终模型中使用此阈值，因此选择不同的阈值，
#具体取决于您愿意容忍的误报数量。因此，平均精度得分为您提供了所有不同阈值选择的平均精度。

#绘画训练集PR曲线
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
precision_1, recall_1, thresholds = precision_recall_curve(y_train, y_trainprba)
plt.step(recall_1, precision_1, color='darkorange', alpha=0.2,where='post')
plt.fill_between(recall_1, precision_1, step='post', alpha=0.2,color='darkorange')
plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall_1,precision_1)
plt.show()
AP_train = average_precision_score(y_train, y_trainprba, average='macro', pos_label=1, sample_weight=None)
