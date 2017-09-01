#train... -l 0.00002
echo 'training ...'
./ffm-train -l 0.00002 -k 10 -t 30 -r 0.02 -s 16 -p ../output/test.ffm ../output/train.ffm ../output/model
#predcting
echo 'predicting ...'
#./ffm-predict ../output/test.ffm ../output/model ../output/test.out
echo 'writing ...'
#sudo python ../src/ffm.py
