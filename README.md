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
1. Using Medical Masks Dataset (https://www.kaggle.com/vtech6/medical-masks-dataset), comes from Eden Social Welfare Foundation which contains the pictures of people wearing medical masks along with the labels containing their descriptions

2. Download images and labels, transforming .xml to a .csv file, with header row 'filename', 'label', 'xmax', 'xmin', 'ymax', 'ymin'.
   For example:
   | filename | label | xmax | xmin | ymax | ymin |
   | -------- | :---: | :--: | :--: | :--: | :--: |
   |c1\_1844849.jpg|good|1246|127|1312|227|
   |c1\_1844849.jpg|none|745|889|862|999|
   
3. Split data for train and test, name the file as train.csv / test.csv.
