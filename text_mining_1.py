import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

## load data ##
url = 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_4_Text_classification/Pakistani%20Traffic%20sentiment%20Analysis.csv'
df = pd.read_csv(url)
# Delete the duplicate rows
df = df.drop_duplicates()

# Displaying the instances of each class
df.groupby('Sentiment').describe()

## vectorization ##
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df=20)
# vectorizer = HashingVectorizer(ngram_range=(1, 2), n_features=200)
# vectorizer = TfidfVectorizer(min_df=20, norm='l2', smooth_idf=True, use_idf=True, ngram_range=(1, 1), stop_words='english')

x = df['Text']
y = df['Sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Apply the vectorizer
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

## define models ##
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=0),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(random_state=0),
    'XGBClassifier': XGBClassifier(),
    'SVC': SVC(probability=True)
}

## Define parameter distributions ##
param_distributions = {
    'LogisticRegression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    'XGBClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'SVC': {
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.001, 0.01, 0.1, 1, 10],
        'degree': [2, 3, 4]
    }
}

results = {}
for model_name, model in models.items():
    print(f"Processing {model_name}")
    param_dist = param_distributions[model_name]
    search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5, random_state=0, scoring='accuracy')
    search.fit(x_train_vectorized, y_train)
    results[model_name] = search.best_estimator_

# Evaluate each best model
for model_name, model in results.items():
    print(f"Evaluating {model_name}")
    accuracy = accuracy_score(y_test, model.predict(x_test_vectorized))
    print(f'The accuracy of the {model_name} model is: ', accuracy)
    ConfusionMatrixDisplay.from_estimator(
        model, 
        x_test_vectorized,
        y_test,
        display_labels=['Positive', 'Negative'],
        cmap='Blues',
        xticks_rotation='vertical'
    )
