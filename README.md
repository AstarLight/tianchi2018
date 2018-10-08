# tianchi2018

## Setup
先配置好Anaconda3来安装Python3.5环境，tensorflow后端
```
source activate python35env
conda install -c conda-forge keras=2.0.1
conda install scikit-learn
conda install matplotlib
pip install imutils
```
## Train
```
python train.py --dataset_train /home/ljs/tianchi2018/data_all1 --model lenet_100.model
```

## Predict
注意训练的model的norm_size大小要和predict里面的一致
```
norm_size = 128
```
```
python predict.py --model lenet_100.model -dtest /home/ljs/tianchi2018/dataset/guangdong_round1_test_a_20180916

```
## create new data -- adjust_data.py
> load_data方法中 分割改成_
	label_name = imagePath.split(os.path.sep)[-1].split(".")[0].split("_")[0]
```
python adjust_data.py --dataPath /Users/cong/tianchi2018/data_all2 --number 200 --outPath /Users/cong/tianchi2018/data_new
# number 是每一类要的数量，我这里默认正常类为两倍number
# dataPath 是原始数据的位置
# outPath 是生成的数据的位置
```
