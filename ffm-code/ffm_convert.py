#encoding: utf-8
import csv

fullstackdict=dict()

#the data saved witg csv
train=csv.DictReader(open('TrainFeature21-24.csv'))

test=csv.DictReader(open('TrainFeature25.csv'))
#features list
fn=['connectionType', 'telecomsOperator', 'creativeID', 'adID', 'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'positionID', 'sitesetID', 'positionType', 'appCategory']
print (fn)
i=0

for rec in train:
    for item in fn:
        k=item+'#'+rec[item]
        if fullstackdict.__contains__(k):
            continue
        else:
            fullstackdict[k]=i
            i+=1
print ('proc tr')
for rec in test:
    for item in fn:
        k=item+'#'+rec[item]
        if fullstackdict.__contains__(k):
            continue
        else:
            fullstackdict[k]=i
            i+=1 
print ('proc te')

trw=open('tttr.ffm','w')
for rec in train:
    string=rec['label']
    for ix,item in enumerate(fn):
        k=item+'#'+rec[item]
        string+=' '+str(ix)+':'+fullstackdict[k]+':1'
    trw.write(string+'\n')
print ('wtr tr')
  
tew=open('ttte.ffm','w')
for rec in train:
    string=rec['label']
    for ix,item in enumerate(fn):
        k=item+'#'+rec[item]
        string+=' '+str(ix)+':'+fullstackdict[k]+':1'
    tew.write(string+'\n')
print ('wtr te')
