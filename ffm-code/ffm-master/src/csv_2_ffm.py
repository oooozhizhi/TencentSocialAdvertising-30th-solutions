# -*- encoding:utf-8 -*-
import collections
from csv import DictReader
from datetime import datetime

train_path = '../data_ori/train.csv'
test_path = '../data_ori/test.csv'
train_ffm = '../output/train.ffm'
test_ffm = '../output/test.ffm'

field = ['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10',\
        'T11','T12','T13','T14','T15','T16','T17','T18','T19','T20',\
        'T21','T22','T23','T24','T25','T26','T27','T28','T29','T30',\
        'T31','T32','T33','T34','T35','T36','T37','T38','T39',\
        'T40','T41','T42','T43','T44','T45','T46','T47','T48','T49',\
        'T50','T51','T52','T53','T54','T55','T56','T57','T58','T59',\
        'T60','T61','T62','T63','T64','T65','T66','T67','T68','T69',\
        'T70','T71','T72','T73','T74','T75','T76','T77','T78','T79',\
        'T80','T81','T82','T83','T84','T85','T86','T87','T88','T89',\
        'T90','T91','T92','T93','T94','T95','T96','T97','T98','T99',\
        'T100','T101','T102','T103','T104','T105','T106','T107','T108','T109','T110']
"""
        ,\
        'T111','T112','T113','T114','T115','T116','T117','T118','T119','T120',\
        'T121','T122','T123','T124','T125','T126','T127','T128','T129','T130',\
        'T131','T132','T133','T134','T135','T136','T137','T138','T139',\
        'T140','T141','T142','T143','T144','T145','T146','T147','T148','T149',\
        'T150','T151','T152','T153','T154','T155','T156','T157','T158','T159',\
        'T160','T161','T162','T163','T164','T165','T166','T167','T168','T169',\
        'T170','T171','T172','T173','T174','T175','T176','T177','T178','T179',\
        'T180','T181','T182','T183','T184','T185','T186','T187','T188','T189',\
        'T190','T191','T192','T193','T194','T195','T196','T197','T198',\
        ]

"""
"""       
        'time_delta_user_creative_next','time_delta_user_creative','time_delta_user_next',\
        'time_delta_user','user_install_app_count_before','user_install_count_previous15','rate_userID_sitesetID',\
        'app_install_count_previous15','count_today_userID_appID','rate_positionID','rate_creativeID','rate_camgaignID',\
        'rate_appID_positionID_connectionType','rate_age_gender_appID','appID_age_tfidf',\

        'rank_user_creative_click','rank_user_creative_click','positionID','positionType',\
        'sitesetID','appID','advertiserID','creativeID','adID','camgaignID','connectionType','telecomsOperator',\
        'clickTime_HH','gender','age','haveBaby','marriageStatus',\
        'hometown_1','residence_1'] 
"""
"""
field = ['clickTime_HH', 'time_delta_user', 'time_delta_user_creative', 'time_delta_user_next', \
        'time_delta_user_creative_next', 'rank_user_click', 'rank_user_creative_click',\
        'user_install_app_count_before', 'app_install_user_count_before', 'user_install_count_previous15', \
        'app_install_count_previous15', 'user_app_before_has_installed', 'creativeID', 'positionID',\
        'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'positionType', 'appCategory_1',\
        'acvertiser_conversion_time_average', 'app_conversion_time_average', 'userID_appCategory_1_tfidf', \
        'userID_appCategory_tfidf', 'appID_age_tfidf', 'appID_gender_tfidf', 'appID_education_tfidf', \
        'appID_haveBaby_tfidf', 'appID_marriageStatus_tfidf', 'appID_hometown_1_tfidf', 'appID_residence_1_tfidf',\
        'count_today_userID', 'count_today_creativeID', 'count_today_appID', 'count_today_userID_appID',\
        'count_today_userID_positionID', 'count_today_camgaignID_connectionType', 'active_appID_positionID_connectionType',\
        'count_appID_positionID_connectionType', 'rate_appID_positionID_connectionType', 'active_positionID_positionType',\
        'count_positionID_positionType', 'rate_positionID_positionType', 'active_positionID_advertiserID',\
        'count_positionID_advertiserID', 'rate_positionID_advertiserID', 'active_positionID_gender', \
        'count_positionID_gender', 'rate_positionID_gender', 'active_hometown_1_residence_1', 'count_hometown_1_residence_1', \
        'rate_hometown_1_residence_1', 'active_gender_education', 'count_gender_education', 'rate_gender_education', \
        'active_positionID_marriageStatus', 'count_positionID_marriageStatus', 'rate_positionID_marriageStatus', \
        'active_age_marriageStatus', 'count_age_marriageStatus', 'rate_age_marriageStatus', 'active_positionID_age',\
        'count_positionID_age', 'rate_positionID_age', 'active_positionID_appID', 'count_positionID_appID', \
        'rate_positionID_appID', 'active_positionID_hometown_1', 'count_positionID_hometown_1', 'rate_positionID_hometown_1',\
        'active_positionID_telecomsOperator', 'count_positionID_telecomsOperator', 'rate_positionID_telecomsOperator', \
        'active_positionID_creativeID', 'count_positionID_creativeID', 'rate_positionID_creativeID', 'active_positionID_education',\
        'count_positionID_education', 'rate_positionID_education', 'active_camgaignID_connectionType', \
        'count_camgaignID_connectionType', 'rate_camgaignID_connectionType', 'active_positionID_connectionType', \
        'count_positionID_connectionType', 'rate_positionID_connectionType', 'active_creativeID_connectionType', \
        'count_creativeID_connectionType', 'rate_creativeID_connectionType', 'active_creativeID_gender', 'count_creativeID_gender',\
        'rate_creativeID_gender', 'active_positionID_camgaignID', 'count_positionID_camgaignID', 'rate_positionID_camgaignID',\
        'active_age_gender', 'count_age_gender', 'rate_age_gender', 'active_camgaignID_age', 'count_camgaignID_age', \
        'rate_camgaignID_age', 'active_adID_connectionType', 'count_adID_connectionType', 'rate_adID_connectionType', \
        'active_camgaignID_gender', 'count_camgaignID_gender', 'rate_camgaignID_gender', 'active_advertiserID_connectionType',\
        'count_advertiserID_connectionType', 'rate_advertiserID_connectionType', 'active_positionID_adID', \
        'count_positionID_adID', 'rate_positionID_adID', 'active_positionID_appCategory_1_appCategory_2',\
        'count_positionID_appCategory_1_appCategory_2', 'rate_positionID_appCategory_1_appCategory_2', \
        'active_positionID_haveBaby', 'count_positionID_haveBaby', 'rate_positionID_haveBaby', \
        'active_residence_1_appCategory_1_appCategory_2', 'count_residence_1_appCategory_1_appCategory_2', \
        'rate_residence_1_appCategory_1_appCategory_2', 'active_appID', 'count_appID', 'rate_appID', \
        'active_advertiserID', 'count_advertiserID', 'rate_advertiserID', 'active_residence_1',\
        'count_residence_1', 'rate_residence_1', 'active_hometown_1', 'count_hometown_1', 'rate_hometown_1',\
        'active_camgaignID', 'count_camgaignID', 'rate_camgaignID', 'active_creativeID', 'count_creativeID',\
        'rate_creativeID', 'active_positionID', 'count_positionID', 'rate_positionID', 'active_appCategory_1_appCategory_2',\
        'count_appCategory_1_appCategory_2', 'rate_appCategory_1_appCategory_2', 'active_advertiserID_appCategory_1_appCategory_2',\
        'count_advertiserID_appCategory_1_appCategory_2', 'rate_advertiserID_appCategory_1_appCategory_2', \
        'active_age_gender_appID', 'count_age_gender_appID', 'rate_age_gender_appID', 'active_hometown_1_residence_1_positionID',\
        'count_hometown_1_residence_1_positionID', 'rate_hometown_1_residence_1_positionID', 'active_clickTime_HH_positionID_connectionType',\
        'count_clickTime_HH_positionID_connectionType', 'rate_clickTime_HH_positionID_connectionType', 'active_clickTime_HH_appID', \
        'count_clickTime_HH_appID', 'rate_clickTime_HH_appID', 'active_clickTime_HH_age', 'count_clickTime_HH_age', \
        'rate_clickTime_HH_age', 'active_clickTime_HH_haveBaby', 'count_clickTime_HH_haveBaby', 'rate_clickTime_HH_haveBaby', \
        'active_clickTime_HH', 'count_clickTime_HH', 'rate_clickTime_HH', 'active_userID_sitesetID', 'count_userID_sitesetID', \
        'rate_userID_sitesetID', 'active_userID_positionID', 'count_userID_positionID', 'rate_userID_positionID', \
        'active_userID_connectionType', 'count_userID_connectionType', 'rate_userID_connectionType', \
        'active_userID_appID', 'count_userID_appID', 'rate_userID_appID', 'active_userID_appPlatform', \
        'count_userID_appPlatform', 'rate_userID_appPlatform']
""" 
#field =['time_delta_user','time_delta_user_creative','time_delta_user_next','time_delta_user_next_creative']             
#field = ['clickTime', 'creativeID', 'userID', 'positionID', 'connectionType', 'telecomsOperator']

table = collections.defaultdict(lambda: 0)


# 为特征名建立编号, filed
def field_index(x):
    index = field.index(x)
    return index

# 为特征值建立编号
def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices

feature_indices = set()

with open(train_ffm, 'w') as fo:
    for e, row in enumerate(DictReader(open(train_path))):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    idx = field_index(k)
                    kv = k + ':' + v
                    features.append('{0}:{1}:1'.format(idx, getIndices(kv)))
                    feature_indices.add(kv + '\t' + str(getIndices(kv)))
        if e % 100000 == 0:
            print(datetime.now(), 'creating train.ffm...', e)
        fo.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))

with open(test_ffm, 'w') as fo:
    for t, row in enumerate(DictReader(open(test_path))):
        features = []
        for k, v in row.items():
            if k in field:
                if len(v) > 0:
                    idx = field_index(k)
                    kv = k + ':' + v
                    if kv + '\t' + str(getIndices(kv)) in feature_indices:
                        features.append('{0}:{1}:1'.format(idx, getIndices(kv)))
        if t % 100000 == 0:
            print(datetime.now(), 'creating validation data and test.ffm...', t)
        fo.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))

fo.close()

