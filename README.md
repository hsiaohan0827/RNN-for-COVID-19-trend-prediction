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
