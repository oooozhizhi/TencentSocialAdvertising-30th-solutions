#train...
sudo ./ffm-train -l 0.00002 -k 8 -t 100 -r 0.02 -s 8 -p ../output/test.ffm  ../output/train.ffm \
      ../out/put/model
#predcting
sudo ./ffm-predict ../output/test.ffm ../output/model ../output/test.out
sudo python ffm.py