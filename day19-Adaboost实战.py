# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:24:34 2022

@author: 11146
"""

#Adaboost属于boosting算法，每一轮都使正确和错误的样本的权重产生变化，不再是RF那种平均投票
#在scikitlearn中adaboost默认是DT模型

import os
os.chdir('C:/Users/11146/Desktop/python-test') ##修改路径
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
#默认ababoost参数跑一下
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(random_state = 0)
classifier.fit(X_train, y_train)
#特有参数：base_estimator、n_estimators、learning_rate、algorithm。
#base_estimator：基础模型，默认是决策树，默认就好；
#n_estimators：基础模型个数，默认50，老朋友了；
#learning_rate：学习率，调整每次叠加模型时的权值，默认1；
#algorithm：两个选项，SAMME和SAMME.R。

y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#结果并没有过拟合
#继续调DT参数
#注意Adaboost和RF不一样，它的DT是这么嵌入的：
#boost = AdaBoostClassifier(DecisionTreeClassifier(), random_state = 0)

#那就先调DT的参数
from sklearn.tree import DecisionTreeClassifier
param_grid=[{
            'max_depth':[50,60,70,80,90,100],
            'min_samples_split':[i for i in range(5,11)],
            'min_samples_leaf':[i for i in range(5,11)],
            },
           ]
boost = DecisionTreeClassifier()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid,n_jobs=-1,verbose=1,cv=10)
grid_search.fit(X_train, y_train)
classifier = grid_search.best_estimator_
classifier.fit(X_train,y_train)
#max_depth=60, min_samples_leaf=5, min_samples_split=9
#继续往下调参
param_grid=[{
            'max_depth':[50,60,70,80,],
            'min_samples_split':[i for i in range(5,8)],
            'min_samples_leaf':[i for i in range(7,11)],
            'max_leaf_nodes':[50,100,150,200,300],
            'ccp_alpha':[0,0.1,0.01,0.001],
            },
           ]
boost = DecisionTreeClassifier()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid,n_jobs=-1,verbose=1,cv=10)
grid_search.fit(X_train, y_train)
classifier = grid_search.best_estimator_
classifier.fit(X_train,y_train)
#ccp_alpha=0.01, max_depth=50, max_leaf_nodes=50,min_samples_leaf=7, min_samples_split=5
#DT最优参数如上
#测试一下结果如何
y_pred = classifier.predict(X_test)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainpred = classifier.predict(X_train)
y_trainprba = classifier.predict_proba(X_train)[:,1]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
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
plt.title('ROC of train')
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
plt.title('ROC of test')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()
#好像很一般
#ccp_alpha=0.01, max_depth=50, max_leaf_nodes=50,min_samples_leaf=7, min_samples_split=5

from sklearn.ensemble import AdaBoostClassifier
boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=50,max_leaf_nodes=50,min_samples_leaf=7,
                                                  min_samples_split=5,ccp_alpha=0.01),random_state = 0)
classifier.fit(X_train, y_train)
#看看结果
y_pred = classifier.predict(X_test)
y_trainpred = classifier.predict(X_train)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

#调adaboost参数
param_grid=[{
            'algorithm':['SAMME','SAMME.R'],
            'n_estimators':[i for i in range(10,60,5)],
            'learning_rate':[i for i in range(1,11)],
            },
           ]
boost = AdaBoostClassifier(random_state = 0)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid,n_jobs=-1,verbose=1,cv=10)
grid_search.fit(X_train, y_train)
classifier = grid_search.best_estimator_
classifier.fit(X_train,y_train)
#特有参数：base_estimator、n_estimators、learning_rate、algorithm。
#base_estimator：基础模型，默认是决策树，默认就好；
#n_estimators：基础模型个数，默认50，老朋友了；
#learning_rate：学习率，调整每次叠加模型时的权值，默认1；
#algorithm：两个选项，SAMME和SAMME.R。

#algorithm='SAMME', learning_rate=1, n_estimators=40
#测试一下
y_pred = classifier.predict(X_test)
y_trainpred = classifier.predict(X_train)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
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
plt.title('ROC of train')
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
plt.title('ROC of test')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

#继续优化一下learning_rate是不是可以小于1
param_grid=[{
            'algorithm':['SAMME','SAMME.R'],
            'n_estimators':[i for i in range(10,60,5)],
            'learning_rate':[0.1,0.2,0.4,0.6,0.8,1.0], 
            },
           ]
boost = AdaBoostClassifier(random_state = 0)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid,n_jobs=-1,verbose=1,cv=10)
grid_search.fit(X_train, y_train)
classifier = grid_search.best_estimator_
classifier.fit(X_train,y_train)
#learning_rate=0.4, n_estimators=15
y_pred = classifier.predict(X_test)
y_trainpred = classifier.predict(X_train)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
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
plt.title('ROC of train')
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
plt.title('ROC of test')
plt.legend(loc="lower right")
#plt.savefig('rf_ljz_training sets muti-ROC.tif',dpi=300)
plt.show()

#不升还降了，看来真的跟演示数据有出入