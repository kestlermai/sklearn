# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:45:57 2023

@author: maihuanzhuo
"""

##不平衡数据
##不平衡数据其实就是对照组和实验组的样本数相差较大，譬如对照组：实验组=1:10或者是1:5，直接影响到模型性能
#因此要对这部分的数据进行处理
#1.下采样（down sampliing）：从多数类中随机抽取部分样本，使其与少数类的数量相同或接近
#2.过采样（over sampling）：从少数类中随机复制或生成新的样本，使其与多数类的数量相同或接近
#3.人工少数类过采样法SMOTE（Synthetic Minority Over-sampling Technique）：利用插值方法在少数类附近产生新的样本点
#4.数据增强（data augmentaion）：对原始数据进行一些变换

#总结一下：
#（1）把少数类的变多：就是用各种算法生成新数据填充进去，个人觉得有点失真，我一般不用这种方法，有兴趣的可以自己试试。
#（2）把多数类的变少：缺点很明显，就是样本量少了。不过个人常用这个方法，具体来说就是倾向性评分匹配，俗称PSM法。
#（3）用集成学习法：也就是使用Xgboost、Catboost等boost字辈的模型，其实他们是有参数来处理不平衡数据的，后面细说。

#在xgboost中有个scale_pos_weight参数
#scale_pos_weight：在样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。通常可以将其设置为负样本的数目与正样本数目的比值。
#“例如，如果负样本数目为1000，正样本数目为100，那么可以将scale_pos_weight设置为10。”，可以先保守一点设置小一点，譬如6,7这样，甚至12-15-20

#Catboost调参，class_weights, auto_class_weights, scale_pos_weight
# class_weights：y标签类别权重。用于类别不均衡处理，默认均为1。
# auto_class_weights：自动计算平衡各类别权重。
# scale_pos_weight：二分类中第1类的权重，默认值1（不可与class_weights、auto_class_weights同时设置）。
#所以跟Xgboost的参数用法大同小异的

#LightGBM调参
#is_unbalance：默认Flase，用于binary分类，如果数据不平衡则设置为True

#RF调参
#class_weight：设置各类别样本的权重，默认是各个样本的权重一致，都为1

#以上都是用算法层面上对模型进行调整，但是可提升的空间很小，想突破上限必须从数据上进行调整。

#倾向匹配得分，PSM全称为Propensity Score Matching
#根据某个特征，从对类别0的1671例中挑出一部分来跟类别1来匹配，形成新的数据集，然后建模。
#其实就是多变少的策略。可以1:1、1:2、1:3等进行匹配
#PSM需要指定匹配的一个或者多个自变量（特征），一般来说，选取人口学特征，比如年龄、性别、民族啥的。也可以根据专业实际情况选别的。

#R 实现方法：
# 安装并加载所需的R包
if (!require(MatchIt)) install.packages('MatchIt')
library(MatchIt)
# 加载数据
data <- read.csv('wwd3.csv', sep = '\t')
# 使用中位数填充缺失值
for (column_name in colnames(data)) {
median_value <- median(data[[column_name]], na.rm = TRUE)
data[[column_name]] <- ifelse(is.na(data[[column_name]]), median_value, data[[column_name]])
}
# 提取自变量和因变量
X_columns <- colnames(data)[-1]
y_column <- 'outcome'
# 使用逻辑回归模型进行倾向性评分计算
ps_model <- glm(outcome ~ ., data = data, family = binomial())
# 保存倾向性评分
data$propensity_score <- ps_model$fitted.values
# 1:3的倾向性评分匹配
matched <- matchit(outcome ~ propensity_score, data = data, method = "nearest", ratio = 3)
# 输出匹配后的数据
matched_data <- match.data(matched)
write.csv(matched_data, 'psm.csv', row.names = FALSE)
# 检查匹配结果
summary(matched)