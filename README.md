# recsys

## 1.Requirements

TensorFlow2.0,Keras, Python3.6, NumPy, sk-learn, Pandas

## 2.Datasets

### 2.1 Criteo

This dataset Contains about 45 million records. There are 13 features taking integer values (mostly count features) and 26 categorical features.
The dataset is available at http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/

在这里我截取一部分数据进行模型训练 data =../data/Criteo/train.txt

### 2.2 Seguro-safe-driver

In the train and test data, features that belong to similar groupings are 
tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, 
feature names include the postfix bin to indicate binary features and 
cat to indicate categorical features. Features without these designations 
are either continuous or ordinal. Values of -1 indicate that the feature was 
missing from the observation. The target columns signifies whether or not a 
claim was filed for that policy holder.

The dataset is available at https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

## 3. 推荐系统实战

### 3.1 第一章.协同过滤

### 3.2 第二章 GBDT+LR

本质上GBDT+LR是一种具有stacking思想的二分类，所以用来解决二分类问题，这个方法出自于Facebook 2014年的论文 Practical Lessons from Predicting Clicks on Ads at Facebook 。
https://zhuanlan.zhihu.com/p/29053940

### 3.3 第三章 MLR

算法简单实现 我们这里只是简单实现一个tensorflow版本的MLR模型
https://www.jianshu.com/p/627fc0d755b2

### 3.4 第四章 DCN

Deep Cross Network模型

https://www.jianshu.com/p/77719fc252fa

https://github.com/Nirvanada/Deep-and-Cross-Keras

https://blog.csdn.net/roguesir/article/details/797632

https://arxiv.org/abs/1708.05123

### 3.5 第五章 PNN

https://github.com/JianzhouZhan/Awesome-RecSystem-Models

https://github.com/Snail110/tensorflow_practice/blob/master/recommendation/Basic-PNN-Demo/PNN.py

https://www.jianshu.com/p/be784ab4abc2