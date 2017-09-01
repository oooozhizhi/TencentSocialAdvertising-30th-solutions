import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import gc
import sys
#drop the fields that we don't want
def drop_feature(train):
    drop_list = ['conversionTime']
    train.drop(drop_list, axis=1, inplace=True)
    return train

#params setting
def paramsSetting():
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.82,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'verbose': 0
    }
    #200
    num_boost_round = 200
    early_stopping_rounds = 5
    return params, num_boost_round, early_stopping_rounds

#predict the ans---------------------------------------------------------------------------
def predict(gbm, test):
    return gbm.predict(test, num_iteration=gbm.best_iteration)

#pre the ratio----------------------------------------------------------------------------
def gbdtFeaGen(lightGBM, data):
    pred = lightGBM.predict(data, num_iteration=lightGBM.best_iteration, pred_leaf=True)
    return pred

#sample the negative ones:
# w is the sample ratio
def sample_0(train, w):
    train_0 = train[train['label'] == 0]
    train_1 = train[train['label'] == 1]
    train_0 = train_0.sample(frac=w)
    train = pd.concat([train_0, train_1], axis=0)
    return train

#train a model with valid and early_stop----------------------------------------------------
def modelTrainWithValid(train, train_label, valid, valid_label, c):
    lgb_train = lgb.Dataset(train.values,
                            label=train_label,
                            feature_name=list(train.columns))
    lgb_eval = lgb.Dataset(valid.values,
                           label=valid_label.values,
                           reference=lgb_train,
                           feature_name=list(train.columns))
    params = paramsSetting()
    gbm = lgb.train(params[0],
                    lgb_train,
                    num_boost_round=params[1],
                    valid_sets=lgb_eval,
                    early_stopping_rounds=params[2]
                    )
    return gbm
#just train a model without valid and early_stop------------------------------------------
def modelTrain(train,train_label):
    lgb_train = lgb.Dataset(train.values,
                            label  = train_label,
                            feature_name = list(train.columns))
    params = paramsSetting()
    #in this situaction i don't need early stop
    gbm = lgb.train(params[0],
                    lgb_train,
                    num_boost_round = params[1])
    return gbm

#save the gbdt treefea---------------------------------------------------------------------
def saveFea(data,fileName):
    #df = pd.DataFrame(data,columns=[x for x in range(len(data))])
    df = pd.DataFrame(data)
    df.to_csv(fileName)

#save the gbdt treefea---------------------------------------------------------------------
def saveDat(data,fileName):
    #df = pd.DataFrame(data,columns=[x for x in range(len(data))])
    print 'writing '+fileName
    data.fillna(0,inplace=True)
    df = pd.DataFrame(data.values,columns=data.columns)
    df.to_csv(fileName)

#wrong : save the gbdt treefea---------------------------------------------------------------------
def saveDatHdf(data,fileName):
    #df = pd.DataFrame(data,columns=[x for x in range(len(data))])
    print 'writing '+fileName
    df = pd.DataFrame(data.values,columns=data.columns)
    df.to_hdf(fileName,'all')
#drop the fields that we don't want--------------------------------------------------------
def dropField(data,getFields,dropFields):
    label = data[getFields]
    data.drop(dropFields,axis=1,inplace=True)
    return label,data

#use GBDT to extract the tree fea-----------------------------------------------------------
def extractFea(train,test,treeNum=100):
    #train set, we have to use all the data,but why the data still 
    #train = train[(train['clickTime'] >= 24000000) & (train['clickTime'] < 30000000)]
    #test set
    print 'drop some fields'
    test.drop(['instanceID', 'clickTime'], axis=1, inplace=True)
    # Train label
    trainLabel,train = dropField(train,'label',['label','clickTime'])
    lightGBM = modelTrain(train, trainLabel)
    print 'generate fea-----'
    train_gbdt_fea = gbdtFeaGen(lightGBM,train)[:,:treeNum]
    test_gbdt_fea = gbdtFeaGen(lightGBM,test)[:,:treeNum]
    print 'check the size of data   train:'+str(len(train_gbdt_fea)) + ' test: ' + str(len(test_gbdt_fea))
    saveFea(train_gbdt_fea,'gbdt_newfea_train_'+str(treeNum)+'.csv')
    saveFea(test_gbdt_fea,'gbdt_newfea_test_'+str(treeNum)+'.csv')
    print 'finish and save data...'

#sample train to valid and train_sample with the ratio---------------------------------------
def sampleValidSet(data,ratio=0.2):
    sampleList = np.random.rand(len(data)) < ratio
    validSet = data[sampleList]
    trainSet = data[~sampleList]
    return trainSet,validSet
#回退的版本先用着吧 train test--------------------------------------------------------------------------------------
def mixModelTrainWithValid(train, test, w):
    # 训练集
    train = train[(train['clickTime'] >= 24000000) & (train['clickTime'] < 30000000)]
    # 测试集
    pred = test[['instanceID']]
    pred['prob'] = 0
    test.drop(['instanceID', 'clickTime'], axis=1, inplace=True)
    repeat_num = 5
    count = 0

    while count < repeat_num:
        # 本次验证集
        msk = np.random.rand(len(train)) < 0.2
        valid = train[msk]
        # 本次训练集
        train_sample = train[~msk]
        # print train_sample
        # train_sample = sample_0(train_sample,w)
        print 'check the:<--', count, '--> Times ', train_sample.shape, valid.shape
        # 验证集label
        valid_label = valid[['label']]
        valid_label['prob'] = 0
        valid.drop(['label', 'clickTime'], axis=1, inplace=True)
        # 训练集label
        train_sample_label = train_sample['label']
        train_sample.drop(['label', 'clickTime'], axis=1, inplace=True)
        gbm = modelTrainWithValid(train_sample, train_sample_label, valid, valid_label['label'], count)
        valid_label['prob'] = predict(gbm, valid)
        print 'check the:<--', count, '-->times logloss: ', log_loss(y_true=valid_label[['label']].values,y_pred=valid_label[['prob']].values)
        print 'it"s offline,mean', valid_label['prob'].mean()
        count+=1
"""
def mixModelTrainWithValidBack(train, test,w,treeNum):
    # train set
    train = train[(train['clickTime'] >= 24000000) & (train['clickTime'] < 30000000)]
    # test set
    pred,test = dropField(test,'instanceID',['instanceID','clickTime'])
    pred['prob'] = 0
    #setting
    routine = 5
    count = 0
    enc = OneHotEncoder()
    while count < routine:
        # valid data,train data sample
        train_sample,valid =sampleValidSet(train,ratio=0.2)
        # why give up the sample???
        # train_sample = sample_0(train_sample,w)
        print 'check the:<--', count, '--> Times ', train_sample.shape, valid.shape
        # valid label
        valid_label,valid = dropField(valid,'label',['label','clickTime'])
        valid_label['prob'] = 0
        # train set label
        train_sample_label,train_sample = dropField(train_sample,'label',['label','clickTime'])

        gbm = modelTrainWithValid(train_sample, train_sample_label, valid, valid_label['label'], count)
        valid_label['prob'] = predict(gbm, valid)
        print 'check the:<--', count, '-->times logloss: ', log_loss(y_true=valid_label[['label']].values,y_pred=valid_label[['prob']].values)
        print 'it"s offline,mean', valid_label['prob'].mean()
        count+=1
"""
#create the pred and drop same fields-------------------------------------------------------------
def dataPre(train,test):
    # train set
    train = train[(train['clickTime'] >= 24000000) & (train['clickTime'] < 30000000)]
    # test set
    pred,test = dropField(test,'instanceID',['instanceID','clickTime'])
    pred['prob'] = 0
    return train,test,pred
#create the train_sample train_sample_label valid valid_label-------------------------------------
def sampleDataPre(train,count):
    # valid data,train data sample
    train_sample,valid =sampleValidSet(train,ratio=0.2)
    # why give up the sample???
    # train_sample = sample_0(train_sample,w)
    print 'check the:<--', count, '--> Times ', train_sample.shape, valid.shape
    # valid label
    valid_label,valid = dropField(valid,'label',['label','clickTime'])
    valid_label['prob'] = 0
    # train set label
    train_sample_label,train_sample = dropField(train_sample,'label',['label','clickTime'])
    return train_sample,train_sample_label,valid,valid_label

#simple lightGBM used to train---------------------------------------------------------------------
def lightGBMTrainWithValid(train,test,w,treeNum):
    train,test,pred = dataPre(train,test)
    #setting
    routine = 5
    count = 0
    enc = OneHotEncoder()
    while count < routine:
        train_sample,train_sample_label,valid,valid_label = sampleDataPre(train,count)
        gbm = modelTrainWithValid(train_sample, train_sample_label, valid, valid_label['label'], count)
        valid_label['prob'] = predict(gbm, valid)
        print 'check the:<--', count, '-->times logloss: ', log_loss(y_true=valid_label[['label']].values,y_pred=valid_label[['prob']].values)
        print 'it"s offline,mean', valid_label['prob'].mean()
        count += 1

# read the hdf data-------------------------------------------------------------------------------
def readDat(fileName):
    print 'reading ' + fileName
    data = pd.read_hdf(fileName)
    return data

# read the csv data--------------------------------------------------------------------------------
def readDatCSV(fileName,size):
    print 'reading ' + fileName
    data = pd.read_csv(fileName,usecols = [x for x in range(1,size)])
    return data

def readDatCSVAll(fileName):
    print 'reading' + fileName
    data = pd.read_csv(fileName)
    return data

def getHead(data):
    print data.columns.tolist()[:]
# run ---------------------------------------------------------------------------------------------
def main():
    #para-----------------------------------------------------------------------
    gbdtList = ['0','1','2','3','4','5','6','7','8','9',\
                '10','11','12','13','14','15','16','17','18','19',\
                '20','21','22','23','24','25','26','27','28','29']
    path = '../mol2/'
    w = 0.05
    #read data-----------------------------------------------------------------------------------
    
    #train = readDat(path + 'train_timego_0621')
    #test = readDat(path + 'test_timego_0621')
    train = readDat('traintemp')
    test  = readDat('testtemp')

    print 'check the data size->train :' + str(len(train)) + ' test :' + str(len(test))
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    gc.collect()
    #function:extract or mixModelTrainWithValid or lightGBM--------------------------------------
    if sys.argv[1] == 'extract':
        print 'begin extract gbdt fea ...'
        extractFea(train,test,treeNum=200)
    elif sys.argv[1] == 'mixModel':
        print 'begin train model ...'
        mixModelTrainWithValid(train,test,w,treeNum=50)
    elif sys.argv[1] == 'lightGBM':
        #import new gbdt data
        newTrain = readDatCSV('gbdt_fea_train_50.csv',51)
        newTest = readDatCSV('gbdt_fea_test_50.csv',51)
        print 'check the data size->train :' + str(len(newTrain)) + ' test :' + str(len(newTest))
        #band the data
        #lightGBMTrainWithValid(newTrain,newTest,w,treeNum=50)
        train = train.loc[~train.index.duplicated(keep='first')]
        newTrain = newTrain.loc[~newTrain.index.duplicated(keep='first')]
        train = pd.concat([train,newTrain],axis=1)
        test = pd.concat([test,newTest],axis = 1)
        print 'begin train model ...'
        lightGBMTrainWithValid(train,test,w,treeNum=50)
if __name__ == '__main__':
    main()

