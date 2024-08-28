import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_2_regression_model/Exercise2BusData.csv'
df = pd.read_csv(url)

# df = pd.read_csv('Exercise2BusData.csv')
df.head(10)
df = df.drop(['Arrival_time','Stop_id','Bus_id','Line_id'], axis=1)
sns.set()
sns.histplot(x=df['Arrival_delay'])
