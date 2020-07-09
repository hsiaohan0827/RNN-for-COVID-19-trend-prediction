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

3.  修改cnn.py中的config
```python
# CONFIG
class TrainConfig
output_dir = 'AlexNet_is32_bs256_ep300_loss'
if not os.path.isdir('CNN_model/'+output_dir):
    os.mkdir('CNN_model/'+output_dir)
logger = SummaryWriter('CNN_log/'+output_dir)

epochs = 500
bch_size = 256
lr = 0.001
imgSize = 32
save_freq = 50
istrain = True
modelPath = 'AlexNet_is32_bs256_ep300_loss+w/ep250.pkl'
```
