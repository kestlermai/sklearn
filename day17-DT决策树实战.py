# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:59:06 2022

@author: 11146
"""

#DT决策树实战


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

#默认参数跑一次
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(X_train, y_train)

#测试结果
y_pred = classifier.predict(X_test)
y_trainpred = classifier.predict(X_train)

#评估结果
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)
#结果明显过拟合，为了得到一致假设而使假设变得过度严格

#纠正过拟合问题，DT提供一些参数来优化过拟合问题：
#max_depth：限制树的最大深度，一般10-100之间。
#min_samples_leaf：默认1，一般数据非常多的情况下可以设置5左右。
#min_samples_split：默认2，一般数据非常多的情况下可以设置5-10。
#min_impurity_decrease：默认0.0，参数适当的设置也可能减小过拟合，我一般不优先调，选默认。

classifier = DecisionTreeClassifier(max_depth=50, min_samples_split=5, min_samples_leaf=5, random_state = 0)
classifier.fit(X_train,y_train)

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


#DT调参
#max_depth：限制树的最大深度，一般10-100之间。
#min_samples_leaf：默认1，一般数据非常多的情况下可以设置5左右。
#min_samples_split：默认2，一般数据非常多的情况下可以设置5-10。
#min_impurity_decrease：默认0.0，参数适当的设置也可能减小过拟合，我一般不优先调，选默认。
#criterion：绝大多数情况下默认gini系数就对了，不用动，效果甚微；
#splitter：两种选择，best或者random；看大佬总结：“实际分类任务设置时默认best就行，
#毕竟random生成的树实在是过于离谱。如果你想对一个数据集有很多种不同的树模型，又不太关注合理性，可以选择random。”
#max_features：特征小于50默认就好；
#max_leaf_nodes：特征不多默认就好；
#class_weight：默认就好；

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

#max_leaf_nodes：整数，None（默认）。最大叶子节点数。如果为None,则无限制。
#ccp_alpha：剪枝时的alpha系数，需要剪枝时设置该参数，默认值是不会剪枝的。
#接下来试着调试max_leaf_nodes跟ccp_alpha的参数
#减少节点为50个
classifier = DecisionTreeClassifier(max_depth=100, min_samples_split=5, min_samples_leaf=5,
                                    max_leaf_nodes=50, random_state = 0)
classifier.fit(X_train,y_train)

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

#训练集跟测试集的AUC都往下掉了
#调alpha值
classifier = DecisionTreeClassifier(max_depth=100, min_samples_split=5, min_samples_leaf=5, 
                                    ccp_alpha=0.001, random_state = 0)
classifier.fit(X_train,y_train)
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

#训练集AUC上升，测试集AUC继续下降，试着调大一点alpha
classifier = DecisionTreeClassifier(max_depth=100, min_samples_split=5, min_samples_leaf=5, 
                                    ccp_alpha=0.01, random_state = 0)
classifier.fit(X_train,y_train)
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

#调大后训练集跟测试集均0.1往下掉


param_grid=[{
            'max_depth':[50,60,70,80,90,100],
            'min_samples_split':[i for i in range(5,11)],
            'min_samples_leaf':[i for i in range(5,11)],
            'max_leaf_nodes':[50,80,100,150,200,250,300], 
            'ccp_alpha':[0.0,0.1,0.001,0.0001],            
            },
           ]
boost = DecisionTreeClassifier()
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(boost, param_grid,n_jobs=-1,verbose=1,cv=10)
grid_search.fit(X_train, y_train)
classifier = grid_search.best_estimator_
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_trainpred = classifier.predict(X_train)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

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

#结果不好，手动调
classifier = DecisionTreeClassifier(max_depth=100, max_leaf_nodes=50,min_samples_split=5, min_samples_leaf=10, 
                                    ccp_alpha=0.001, random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
y_trainpred = classifier.predict(X_train)
y_testprba = classifier.predict_proba(X_test)[:,1]
y_trainprba = classifier.predict_proba(X_train)[:,1]
cm_test = confusion_matrix(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_trainpred)
print(cm_train)
print(cm_test)

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
#结果不错，训练集auc=0.9438，测试集auc=0.8178，后续放spsspro看看