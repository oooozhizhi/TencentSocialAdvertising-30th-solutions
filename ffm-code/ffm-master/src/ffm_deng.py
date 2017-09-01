# encoding: utf-8
'''
Created on 2017年6月24日

@author: c
'''
import csv
def run(infile,outfile):
    fullstackdict=dict()
    # 分别读进来train和test，
    train=csv.DictReader(open(infile))

    fn = ['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10',\
        'T11','T12','T13','T14','T15','T16','T17','T18','T19','T20',\
        'T21','T22','T23','T24','T25','T26','T27','T28','T29','T30',\
        
        'time_delta_user_creative_next','time_delta_user_creative','time_delta_user_next',\
        'time_delta_user','user_install_app_count_before','user_install_count_previous15','rate_userID_sitesetID',\
        'app_install_count_previous15','count_today_userID_appID','rate_positionID','rate_creativeID','rate_camgaignID',\
        'rate_appID_positionID_connectionType','rate_age_gender_appID','appID_age_tfidf',\

        'rank_user_creative_click','rank_user_creative_click','positionID','positionType',\
        'sitesetID','appID','advertiserID','creativeID','adID','camgaignID','connectionType','telecomsOperator',\
        'clickTime_HH','gender','age','haveBaby','marriageStatus',\
        'hometown_1','residence_1']
    #fn=['connectionType', 'telecomsOperator', 'creativeID', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'positionID', 'sitesetID', 'positionType', 'appCategory']
    #fn=['connectionType']
    print (fn)
    i=0

    for rec in train:
        # print rec
        for item in fn:
            k=item+'#'+rec[item]
            if fullstackdict.__contains__(k):
                continue
            else:
                fullstackdict[k]=i
                i+=1
                #print i
    print ('proc tr')

    train=csv.DictReader(open(infile))
    print 'gg'
    trw=open(outfile,'w')
    print train
    for rec in train:
        string=rec['label']
        for ix,item in enumerate(fn):
            k=item+'#'+rec[item]
            string+=' '+str(ix)+':'+str(fullstackdict[k])+':1'
        trw.write(string+'\n')
    print ('wtr tr')
      


run('../data_ori/train.csv','../output/train.ffm')

run('../data_ori/test.csv','../output/test.ffm')
