import pandas as pd
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import os

##  load data  ##
url = 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_4_Text_classification/Pakistani%20Traffic%20sentiment%20Analysis.csv'
df = pd.read_csv(url)

##  vectorization  ##

