#train...
sudo ../libffm-1.13/ffm-train -l 0.00002 -k 8 -t 100 -r 0.02 -s 8 -p ../output/test.ffm  ../output/train.ffm \
      ../output/model
#predcting
sudo ../libffm-1.13/ffm-predict ../output/test.ffm ../output/model ../output/test.out
sudo python ffm.py