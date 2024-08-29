'''
This is first assignment of Zhenghao Liu
This script performs regression analysis using XGBoost to predict bus arrival delays.
It includes data preprocessing, model training, hyperparameter tuning with GridSearchCV, and evaluation.
'''
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

URL = 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_2_regression_model/Exercise2BusData.csv'
df = pd.read_csv(URL)
df = shuffle(df,random_state=1)
df = df.iloc[:1000]
df = df.drop(['Arrival_time','Stop_id','Bus_id','Line_id'], axis=1)
x = df.drop(['Arrival_delay'], axis=1)
y = df['Arrival_delay']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', nthread=4)

param_grid = {
    'max_depth': np.arange(3,10,1),  # Maximum depth of trees
    'eta': np.arange(0.3,0.1-0.05,-0.05),  # Learning rate
    'n_estimators': np.arange(100,1000,100)  #Number of boosting rounds
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=5, verbose=1)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score (RMSE): {-grid_search.best_score_}')

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'Root Mean Squared Error (RMSE): {rmse}')

xgb.plot_importance(best_model)
plt.title('Feature Importance')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', lw=2)
plt.show()
