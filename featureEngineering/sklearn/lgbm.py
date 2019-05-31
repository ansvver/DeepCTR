import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import lightgbm as lgb

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import gc

def main():

    print('Load data...')
    localPath='/data/code/DeepCTR/data/dataForSkearn/'
    df_train = pd.read_csv(localPath+'train.csv')
    # df_test = pd.read_csv(localPath+'test.csv')   #测试集中的like，finish都为-1，不能用于验证模型好坏

    #将df_train划分为训练集和验证集
    trainData,valData=train_test_split(
            df_train,test_size=0.1,random_state=0,stratify=df_train['like'])

    del df_train
    gc.collect()
    NUMERIC_COLS = ['item_id','uid','channel','duration_time','item_pub_hour','device_Cnt_bin','authorid_Cnt_bin','musicid_Cnt_bin','uid_playCnt_bin',\
    'itemid_playCnt_bin','user_city_score_bin','item_city_score_bin','title_topic','gender','beauty','relative_position_0',\
    'relative_position_1','relative_position_2','relative_position_3']



    y_train = trainData['like']  # training label
    y_val = valData['like']


    X_train = trainData[NUMERIC_COLS]  # training dataset
    X_val = valData[NUMERIC_COLS]  # testing dataset

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'metric': {'binary_logloss'},
    #     'num_leaves': 64,
    #     'num_trees': 50,
    #     'learning_rate': 0.001,   #小一些，accuracy高
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.9,  #每次迭代时用的数据比例，加快训练速度和减小过拟合
    #     'bagging_freq': 5,
    #     'verbose': 0,
    #     'save_binary': 'true'
    # }

    params ={'boosting_type': 'gbdt',
             'num_leaves': 60,
             'max_depth':6,
             'num_trees':20,
             'metric': {'binary_logloss','auc'},
             'learning_rate': 0.007286914891341039,
             'min_child_samples': 90,
             'reg_alpha': 0.9591836734693877,
             'reg_lambda': 0.8775510204081632,
             'colsample_bytree': 0.6888888888888889,
             'subsample': 0.5151515151515151}
    # 该参数下
    # Normalized Cross Entropy 2.40653015432871
    # auc is 0.6842660756576155

    num_leaf = 60

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,            #训练数据集
                    num_boost_round=100,  #迭代次数，通常100+
                    early_stopping_rounds=50,
                    valid_sets=lgb_eval)   #验证数据集

    # Starting from the 2.1.2 version, default value for the "boost_from_average" parameter in "binary" objective is true.
    # This may cause significantly different results comparing to the previous versions of LightGBM.
    # Try to set boost_from_average=false, if your old models produce bad results

    print('Save model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Start predicting...')
    print("用predict函数来测试模型效果")

    y_pred_binary = gbm.predict(X_train)
    # print('不设置pred_leaf=True，则输出的是概率结果')
    # print(np.array(y_pred_binary).shape)
    # print(y_pred_binary[:10])

    # predict and get data on leaves, training data
    y_pred = gbm.predict(X_train, pred_leaf=True) #这里的y_pred是train对应的
    print("设置pred_leaf=True，输出每棵树的叶子节点编号")
    print(np.array(y_pred).shape)  #(19622340, 100)  因为设置了100棵树？输出的是每棵树的叶子节点编号
    print(y_pred[:10])

    print('Writing transformed training data')
    # print(len(y_pred))    #19622340
    # print(len(y_pred[0])) #100


    train_row=[]
    train_col=[]
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
        # print(temp)
        train_row.extend([i]*len(y_pred[0]))
        train_col.extend(temp)

        # transformed_training_matrix[i][temp] += 1
    train_data=[1]*len(y_pred)*len(y_pred[0])
    transformed_training_matrix=sparse.csc_matrix((train_data, (train_row, train_col)), shape=(len(y_pred), len(y_pred[0]) * num_leaf),dtype=np.int8)
    # print(transformed_training_matrix.toarray()[:10])  #toarray会消耗很多内存


    y_pred = gbm.predict(X_val, pred_leaf=True)   #这里的y_pred是test对应的
    print('Writing transformed testing data')
    # transformed_testing_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf], dtype=np.int32)
    # transformed_testing_matrix=sparse.csc_matrix((len(y_pred), len(y_pred[0]) * num_leaf), dtype=np.int8)
    test_row=[]
    test_col=[]
    for i in range(0, len(y_pred)):
        temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])  #y_pred输出的是每棵树的叶子节点的编号
        test_row.extend([i]*len(y_pred[0]))
        test_col.extend(temp)
        # transformed_testing_matrix[i][temp] += 1
    test_data=[1]*len(y_pred)*len(y_pred[0])
    transformed_testing_matrix=sparse.csc_matrix((test_data, (test_row, test_col)), shape=(len(y_pred), len(y_pred[0]) * num_leaf),dtype=np.int8)
    # print(transformed_testing_matrix.toarray()[:10])


    lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
    lm.fit(transformed_training_matrix,y_train)  # fitting the data
    y_pred_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label

    print(y_pred_test)  #预测的是一个概率

    #自定义交叉熵，用来求目标与预测值之间的差距
    NE = (-1) / len(y_pred_test) * sum(((1+y_val)/2 * np.log(y_pred_test[:,1]) +  (1-y_val)/2 * np.log(1 - y_pred_test[:,1])))
    print("Normalized Cross Entropy " + str(NE))
    #Normalized Cross Entropy 0.015235707048187584   20棵树


    # precision = precision_score(y_val, y_pred_test[:,1])
    # recall = recall_score(y_val, y_pred_test[:,1])
    # accuracy = accuracy_score(y_val, y_pred_test[:,1])
    # F1_Score = f1_score(y_val, y_pred_test[:,1])


    # 产生fpr,tpr用于画ROC曲线
    fpr,tpr,_ = roc_curve(y_val, y_pred_test[:,1])    #ValueError: bad input shape (2761799, 2)
    roc_auc = roc_auc_score(y_val,y_pred_test[:,1])
    print('auc is {}'.format(roc_auc))


    # 开始画图
    plt.figure()
    lw = 2

    plt.plot(fpr, tpr, color='orange',
              lw=lw, label='ROC_Curve (area = %0.2f)'% roc_auc)

    plt.plot([0,1],[0,1], color='navy',lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Line')
    plt.legend(loc='lower right')
    plt.show()



if __name__ == '__main__':
    main()