# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 20:58:37 2022

@author: 11146
"""

#KNN实战调参

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


#利用网格搜索（Grid Search）的方法，高效寻找参数
#在R中做RF调参时，利用Grid Search寻找最佳mtry
#n_neighbors（K值，默认邻居的数目，取值为奇数）
#weights（是否考虑权重，uniform统一权重；distance
#p值（距离类型，只有当 weights = 'distance' 时，p才有意义；P=1，曼哈顿距离L1；P=2，欧拉距离L2；3，其他）

from sklearn.neighbors import KNeighborsClassifier
param_grid=[
            {
             'weights':['uniform'],
             'n_neighbors':[3,5,7,9,11]
            },
            {
             'weights':['distance'],
             'n_neighbors':[3,5,7,9,11],
             'p':[i for i in range(1,4)]
            },       
            ]
#先试weight=uniform时，K值取值为3、5、7、9;然后再试weight=distance时的K值，P值取1、2、3，range（1,4）=循环到3
boost = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid, scoring='roc_auc', n_jobs = -1, verbose = 1)   

#class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, 
            #n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch=‘2*n_jobs’, 
            #error_score=’raise’, return_train_score=’warn’)
#estimator：基本模型,KNN、RF等等模型
#param_grid：需要最优化的参数的取值范围，可以是列表或者字典；
#scoring：模型评价标准。分为几种情况：
#默认None，则使用estimator的误差估计函数；
#使用一种score，则需要指定用哪一种，比如使用scoring=‘roc_auc’；可选的有一堆：‘accuracy’，‘balanced_accuracy’，‘roc_auc’等，
#具体看这里：https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#cv：交叉验证参数，默认五折交叉验证。

#构建完模型后需要fit
grid_search.fit(X_train,y_train)

#Fit结束后，有三个函数可以用：
#调用 .best_estimator_ ，可以输出最佳模型；
#调用 .best_score_ ， 可以输出最佳模型对应的度量；
#调用 .best_params_ ，可以输出最佳模型对应的参数；
classifier = grid_search.best_estimator_
classifier.fit(X_train, y_train)

#得出最佳参数

#预测结果
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]

y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#输出的结果过拟合了

#绘画测试集混淆矩阵
classes = list(set(y_test))
classes.sort()
plt.imshow(cm_test, cmap=plt.cm.Blues)
indices = range(len(cm_test))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('guess')
plt.ylabel('fact')
for first_index in range(len(cm_test)):
    for second_index in range(len(cm_test[first_index])):
        plt.text(first_index, second_index, cm_test[first_index][second_index])
 
plt.show() 

#绘画训练集混淆矩阵
classes = list(set(y_train))
classes.sort()
plt.imshow(cm_train, cmap=plt.cm.Blues)
indices = range(len(cm_train))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('guess')
plt.ylabel('fact')
for first_index in range(len(cm_train)):
    for second_index in range(len(cm_train[first_index])):
        plt.text(first_index, second_index, cm_train[first_index][second_index])
 
plt.show()
 
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
plt.plot([0, 1], [0, 1], '--', color='navy')    
plt.plot(fpr_train, tpr_train, 'k--',label='Mean ROC (area = {0:.4f})'.format(auc_train), lw=2,color='darkorange')
plt.xlim([-0.01, 1.01])     
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')    
plt.title('Please replace your title')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
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
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

#绘画测试集PR曲线
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
precision_1, recall_1, thresholds = precision_recall_curve(y_test, y_testprba)
plt.step(recall_1, precision_1, color='darkorange', alpha=0.2,where='post')
plt.fill_between(recall_1, precision_1, step='post', alpha=0.2,color='darkorange')
plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall_1,precision_1)
plt.show()
AP_test = average_precision_score(y_test, y_testprba, average='macro', pos_label=1, sample_weight=None)

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