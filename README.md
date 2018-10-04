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
```
python predict.py --model lenet_100.model -dtest /home/ljs/tianchi2018/dataset/guangdong_round1_test_a_20180916

```
