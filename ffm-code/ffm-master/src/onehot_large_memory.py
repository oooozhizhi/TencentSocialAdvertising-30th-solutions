import pandas as pd
#import constants
import math
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import multiprocessing
from sklearn.datasets import dump_svmlight_file
import datetime
import io
import gc
#from util import utils
import utils as utils
import sys
import os

def map_time_delta(x):
    if x == -1:
        return 0
    if x < 300:
        return 1
    if x < 900:
        return 2
    if x < 7200:
        return 3
    if x < 75600:
        return 4
    return 5


def logten(x):
    if x >= 10:
        return int(math.log(x))
    return 0


def logtwo(x):
    if x >=2 :
        return int(math.log(x, 2))
    return 0


def map_rate(x):
    if x == -1:
        return 0
    return int(x*1000)


def map_tfidf(x):
    return int(x*100)


def map_today_count(x):
    if x >= 10:
        # x /= 10
        return int(math.log(x))
    return 0


def map_count(x):
    if x == -1:
        return 0
    return int(x*10)

# build train test data
def build_train_test(train_f, test_f):
    test = pd.read_csv(test_f)
    train = pd.read_csv(train_f)
    #test = pd.read_hdf(test_f)
    #train = pd.read_hdf(train_f)
    len_train = len(train)
    print train.shape, test.shape
    test.rename(columns={'instanceID': 'label'}, inplace=True)
    train_test = pd.concat([train, test], axis=0)
    train_test['label'].to_hdf('../output/train_test_y', 'all')
    y_series = train_test['label']
    y = train_test['label'].as_matrix()
    train_test.drop('label', axis=1, inplace=True)
    del train, test
#     print y[:10]
    return train_test, y_series, y, len_train


# dump revised column to file
def revise_col((df, cols, func)):
    # print cols
    for col in cols:
        s = df[col].apply(lambda x: func(x))
        sdf = s.to_frame()
        sdf.columns = [col]
        sdf.to_hdf(constants.other_work_dir+'/temp_col_bb/'+col,  'all', complevel=2, complib='blosc')
        del sdf, s

# join int columns
def load_col(train_test, apply_cols):
    processNum = 23
    pool = multiprocessing.Pool(processes=processNum)
    results = pool.map(process_col, [col for col in apply_cols])
    pool.close()
    pool.join()

    for rst in results:
        col = rst[0]
        train_test[col] = rst[1]
        print col
    del results


def process_col(col):
    ''' Process one file: count number of lines and words '''
    df = pd.read_hdf(constants.other_work_dir + '/temp_col_bb/' + col)
    # print col
    return col, df

# revise float to int for split bucket
def revise_data(train_test):
    print "revise starts."
    count_cols = [col for col in train_test.columns if col.startswith('active')]
    rate_cols = [col for col in train_test.columns if col.startswith('rate')]
    time_cols = ['time_delta_user', 'time_delta_user_creative', 'time_delta_user_next', 'time_delta_user_creative_next']
    app_install_cols = ['app_install_user_count_before', 'app_install_count_previous15']
    logtwo_cols = ['user_install_app_count_before', 'acvertiser_conversion_time_average', 'app_conversion_time_average']
    tf_idf_cols = ['userID_appCategory_tfidf', 'appID_age_tfidf', 'appID_gender_tfidf', 'appID_residence_1_tfidf']
    count_today_cols = ['count_today_creativeID', 'count_today_appID', 'count_today_camgaignID_connectionType']
    # count_cols = [item for item in count_cols if item not in count_today_cols]

    # print train_test.iloc[:2, :]
    train_test.fillna(0)

    count_split = len(count_cols) / 12
    rate_split = len(rate_cols) / 6
    processNum = 23
    pool = multiprocessing.Pool(processes=processNum)
    for i in xrange(12):
        if i != 11:
            col_list = count_cols[i*count_split:(i+1)*count_split]
            pool.apply_async(revise_col, ([train_test[col_list], col_list, map_count], ))
        else:
            col_list = count_cols[i*count_split:]
            pool.apply_async(revise_col, ([train_test[col_list], col_list, map_count], ))

    for i in xrange(6):
        if i != 5:
            col_list = rate_cols[i*rate_split:(i+1)*rate_split]
            pool.apply_async(revise_col, ([train_test[col_list], col_list, map_rate], ))
        else:
            col_list = rate_cols[i*rate_split:]
            pool.apply_async(revise_col, ([train_test[col_list], col_list, map_rate], ))

    pool.apply_async(revise_col, ([train_test[time_cols], time_cols, map_time_delta], ))
    pool.apply_async(revise_col, ([train_test[app_install_cols], app_install_cols, logten], ))
    pool.apply_async(revise_col, ([train_test[logtwo_cols], logtwo_cols, logtwo], ))
    pool.apply_async(revise_col, ([train_test[tf_idf_cols], tf_idf_cols, map_tfidf], ))
    pool.apply_async(revise_col, ([train_test[count_today_cols], count_today_cols, map_today_count], ))
    # print train_test.iloc[:2, :]

    pool.close()
    pool.join()

    apply_cols = count_cols + rate_cols + time_cols + app_install_cols + logtwo_cols + tf_idf_cols + count_today_cols
    load_col(train_test, apply_cols)
    # train_test.to_hdf(constants.other_work_dir+'/train_test_revise', 'all')
    print train_test.head(2)['count_today_appID']

'''
set field operations
'''


def process_ffm(col):
    ''' Process one file: count number of lines and words '''
    df = pd.read_hdf(constants.other_work_dir + '/temp_ffm/' + col)
    return col, df


# join ffm columns
def load_ffm(train_test):
    processNum = 23
    pool = multiprocessing.Pool(processes=processNum)
    results = pool.map(process_ffm, [col for col in train_test.columns])
    pool.close()
    pool.join()

    for rst in results:
        col = rst[0]
        train_test[col] = rst[1]
        col_file = constants.other_work_dir + '/temp_col_bb/' + col
        if os.path.isfile(col_file):
            os.remove(col_file)
            print col
        fm_file = constants.other_work_dir + '/temp_ffm/' + col
        # os.remove(fm_file)
    del results


# change line break
def windows2linux(src_file, to_file):
    to = io.open(to_file, 'w')
    with open(src_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            to.write(unicode(line) + unicode('\n'))
    to.close()


# error in other thread won't display in the console
def onehot_col((df, cols, col_idx_dict, field_dict, offset_dict)):
    # print cols
    for col in cols:
        s = df[col].apply(
            lambda x: map_idx(x, col_idx_dict[col], offset_dict[col], field_dict[col]))
        sdf = s.to_frame()
        sdf.columns = [col]
        sdf.to_hdf(constants.other_work_dir + '/temp_ffm/' + col, 'all')
        del sdf, s

# convert feature field to ffm format
def map_idx(x, col_dict, offset, field):
    idx = col_dict[x]
    return str(field) + ':' + str(offset + idx) + ':1'

# build ffm feature file
def build_ffm_hand(train_test):
    print "Building ffm starts."
    col_idx_dict = {}
    for col in train_test.columns:
        vals = train_test[col].unique().tolist()
        col_idx_dict[col] = utils.list2dict(vals)

    processNum = 23
    cols = train_test.columns
    field_dict = utils.list2dict(cols)
    offset_dict = {}
    offset = 0
    for col in train_test.columns:
        offset_dict[col] = offset
        offset += len(col_idx_dict[col])

    split_size = len(cols) / 23
    pool = multiprocessing.Pool(processes=processNum)
    for i in xrange(23):
        if i != 22:
            col_list = cols[i*split_size:(i+1)*split_size]
            cid = {key: col_idx_dict[key] for key in col_idx_dict.keys() if key in col_list}
            fd = {key: field_dict[key] for key in field_dict.keys() if key in col_list}
            od = {key: offset_dict[key] for key in offset_dict.keys() if key in col_list}
            pool.apply_async(onehot_col, ([train_test[col_list], col_list, cid, fd, od], ))
        else:
            col_list = cols[i*split_size:]
            cid = {key: col_idx_dict[key] for key in col_idx_dict.keys() if key in col_list}
            fd = {key: field_dict[key] for key in field_dict.keys() if key in col_list}
            od = {key: offset_dict[key] for key in offset_dict.keys() if key in col_list}
            pool.apply_async(onehot_col, ([train_test[col_list], col_list, cid, fd, od], ))

    pool.close()
    pool.join()
    load_ffm(train_test)

if __name__ == '__main__':
    
    start = datetime.datetime.now()
    print start
    train_file = '../data_ori/train.csv'
    test_file ='../data_ori/test.csv'
    train_test, y_series, y, len_train = build_train_test(train_file, test_file)
    #we don't need this , we have done it.
    #revise_data(train_test)
    end1 = datetime.datetime.now()
    print "revise time: " + str(end1 - start)
    gc.collect()

    build_ffm_hand(train_test)
    train_test.insert(0, 'label', y_series)
    train_test.iloc[:len_train, :].to_csv('../output/train.ffm', sep=' ',header=None, index=False)
    train_test.iloc[len_train:, :].to_csv('../output/test.ffm', sep=' ',header=None, index=False)

    # no need on linux
    # windows2linux(constants.other_work_dir+'/ffm/train_sa.ffm', constants.other_work_dir+'/ffm/train_lf_sa.ffm')
    # windows2linux(constants.other_work_dir + '/ffm/test_sa.ffm', constants.other_work_dir + '/ffm/test_lf_sa.ffm')
    print "ffm time: " + str(datetime.datetime.now()-end1)
