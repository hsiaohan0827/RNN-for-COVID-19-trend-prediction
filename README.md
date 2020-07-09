# RNN-for-COVID-19-trend-prediction
implement a simple RNN model for predicting trend of confirmed people of COVID-19 using pytorch


## Setup
1. 創建一個新環境
```
python3 -m venv env_name
```
2. activate environment
```
source env_name/bin/activate
```
3. 安裝requirement.txt中的套件
```
pip3 install -r requirements.txt
```


## Download Data
1. Using COVID-19 Data from 
CSSEGISandData (https://github.com/CSSEGISandData/
COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv), which contains global daily comfirmed number of COVID-19.
   
2. We only utilize the Country/Region-wise data, so integrate the Province/State-wise data into same row. i.e. each Country/Region occupy only one row.
   For example:
   | Country/Rigion | Lat | Long | date1 | date2 | date3 | ... |
   | -------- | :---: | :--: | :--: | :--: | :--: | :--: |
   |Afghanistan|33.0|65.0|0|0|0| |
   |Albania|41.1533|20.1683|0|0|0| |


## Training
1. set correlation coefficient threshold in line 50 to filter the country data.

2.  run preprocess.py for making daily difference sequence as input.  
```
python3 preprocess.py
```

3.  修改config/config.py中的config
```python
# CONFIG
class RNNConfig:
    model_type = 'LSTM' 
    input_dim = 1
    hidden_dim = 32
    layer_dim = 1
    dropout = 0
    
class TrainConfig
    isTrain = False
    interval = 10
    thres = 0
```
## configuration
- **model_type** - RNN type (RNN / LSTM / GRU).
- **input_dim** - RNN input dimension.
- **hidden_dim** - RNN hidden dimension.
- **layer_dim** - RNN layer number.
- **dropout** - dropout for each RNN layer (work when layer_dim > 1)
- **isTrain** - if the model do train or test
- **interval** - the length of the input sequence, i.e 用多長的天數去預測隔天的趨勢
- **thres** - threshold to filter the country. (跟preprocess.py中的threshold一致)

3.  run main.py
```
python3 main.py
```

### tensorboardX
可以使用tensorboard觀察loss及accuracy變化
```
tensorboard --logdir log
```

## Testing
1. 修改istrain及ckpt
```python
class TrainConfig:
   isTrain = False
   ckpt = 'LSTM|L-10|bs-500|in-1|layer-1|hid-32|th-0|Adam lr-0.001'
   epoch = 500
```
2. run main.py, get global trend map for each country.
```
python3 main.py
```
![Covid-19 trend](https://github.com/hsiaohan0827/RNN-for-COVID-19-trend-prediction/blob/master/Covid-19%20trend.png)
