import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results = []
params_df = pd.read_excel()
#-------------------------------------------------data preprocessing------------------------------------------------------------------

# Define the URL of a CSV file containing data.
url = 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_7_Neural_networks/Exercise7data.csv'

# Read the CSV data from the specified URL into a DataFrame (assuming you have the pandas library imported as 'pd').
df = pd.read_csv(url)

# Limit the DataFrame to the first 1000 rows (selecting a subset of the data).
df = df.iloc[:1000]

# Drop specific columns (Arrival_time, Stop_id, Bus_id, Line_id) from the DataFrame.
df = df.drop(['Arrival_time', 'Stop_id', 'Bus_id', 'Line_id'], axis=1)

# Extract the features (input variables) by dropping the 'Arrival_delay' column.
x = df.drop(['Arrival_delay'], axis=1)

# Extract the target variable ('Arrival_delay') as the variable to predict.
y = df['Arrival_delay']

# splite the train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Loop
for index, row in params_df.iterrows():
    if index < 2:  # Skip the first two rows (header and labels)
        continue
    
    # Extract parameters from the current row
    hidden_layer1_size = int(row['Unnamed: 1'])  # Hidden_layer1
    hidden_layer2_size = int(row['Unnamed: 2'])  # Hidden_layer2
    dropout_rate = float(row['Unnamed: 3'])  # Dropout
    activation_function = row['Unnamed: 4']  # Activation Function
    batch_size = int(row['Unnamed: 5'])  # Batch Size
    
    # Create and compile the model based on the parameters
    model = Sequential()
    model.add(Dense(hidden_layer1_size, activation=activation_function, input_dim=X_train.shape[1]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_layer2_size, activation=activation_function))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    # Train the model with early stopping and reduced learning rate
    early_stop = EarlyStopping(monitor='val_mae', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=3)
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=batch_size, 
              callbacks=[early_stop, reduce_lr], verbose=0)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Append results to list (to add them to the dataframe later)
    results.append([mae, mse, r2])

# Add the results back into the dataframe
for i, (mae, mse, r2) in enumerate(results):
    params_df.loc[i+2, 'MAE'] = mae
    params_df.loc[i+2, 'MAPE'] = mse  # Assuming the MSE column needs to store MAPE instead
    params_df.loc[i+2, 'R-Squared'] = r2

# Save the updated dataframe back to the Excel file
output_file_path = '/mnt/data/updated_nureal_network_parameters.xlsx'
params_df.to_excel(output_file_path, index=False)

# Display the updated table with the evaluation metrics
import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Neural Network Parameters", dataframe=params_df)
