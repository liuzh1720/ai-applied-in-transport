import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Load data
data_df = pd.read_csv('C:\\Users\\99508\\OneDrive\\KTH相关\\ai applied in transport\\exercise5_clustering\\dataset_exercise_5_clustering_highway_traffic.csv', sep=";")
nintvals = 288

# Data processing
def datapreprocessing(data_df):
    data_df = data_df.sort_values(["Date", "Interval_5"])
    days = np.unique(data_df['Date'].values)
    ndays = len(days)
    day_subsets_df = data_df.groupby(["Date"])
    vectorized_day_dataset = np.full((ndays, nintvals), np.nan)
    
    for i in range(ndays):
        df_t = day_subsets_df.get_group(days[i])
        for j in range(len(df_t)):
            vectorized_day_dataset[i, df_t.iloc[j]["Interval_5"]] = df_t.iloc[j]["flow"]
    
    # Remove days with any NaNs
    vectorized_day_dataset_no_nans = vectorized_day_dataset[np.all(~np.isnan(vectorized_day_dataset), axis=1)]
    return vectorized_day_dataset_no_nans

# Divide data_df and normalization
processed_data = datapreprocessing(data_df)
scaler = StandardScaler()
processed_data_normalized = scaler.fit_transform(processed_data)
data_df_train, data_df_test = train_test_split(processed_data_normalized, test_size=0.2, random_state=0)

# Ensure that train and test data are in correct format for processing
data_df_train = pd.DataFrame(data_df_train)
data_df_test = pd.DataFrame(data_df_test)

# Find closest centroid
def find_the_closest_centroid(centroids, new_day):
    distances = np.linalg.norm(centroids - new_day, axis=1)
    return np.argmin(distances)

# Calculate external score
def external_score(model, X_train, X_test):
    # Initialize centroids list
    centroids = []

    # Determine cluster centers
    if hasattr(model, 'cluster_centers_'):
        centroids = model.cluster_centers_
    elif hasattr(model, 'means_'):
        centroids = model.means_
    else:
        # Compute centroids manually for models without cluster_centers_
        cluster_labels = model.labels_
        unique_labels = np.unique(cluster_labels)
        
        for i in unique_labels:
            # Skip noise points
            if i == -1:
                continue
            
            indices = np.where(cluster_labels == i)[0]
            if len(indices) > 0:
                centroid = np.mean(X_train[indices], axis=0)
                centroids.append(centroid)
        
        # Convert centroids list to numpy array
        centroids = np.array(centroids)
    
    # Check if there are valid centroids
    if centroids.size == 0:
        raise ValueError("No valid centroids found. Check if all data points are noise.")
    
    # Get cluster labels for the test set
    test_labels = np.array([find_the_closest_centroid(centroids, x) for x in X_test])
    
    # Calculate predicted values and errors
    total_mae = 0
    total_mape = 0
    prediction_counts = 0
    n_past_intervals_for_classification = 5

    for i in range(len(X_test)):
        for j in range(n_past_intervals_for_classification, X_test.shape[1] - 1):
            centroid_index = test_labels[i]
            if centroid_index < len(centroids):
                predicted_value = centroids[centroid_index][j + 1]
                mae_t = abs(predicted_value - X_test[i][j + 1])
                mape_t = abs(predicted_value - X_test[i][j + 1]) / X_test[i][j + 1] if X_test[i][j + 1] != 0 else 0
                total_mae += mae_t
                total_mape += mape_t
                prediction_counts += 1

    # Return the average MAE and MAPE
    return total_mae / prediction_counts, abs(total_mape / prediction_counts)

# Define parameter grids for each model
kmeans_parameters = [3, 4, 5, 6, 7, 8, 9, 10]
agglo_parameters =  [3, 4, 5, 6, 7, 8, 9, 10]
dbscan_eps = [10, 11, 12, 13, 14]
dbscan_min_samples = [5, 10, 15]
gmm_parameters = [3, 4, 5, 6, 7, 8, 9, 10]

# Initialize results list
results = []

# Iterate over KMeans parameters
for n_clusters in kmeans_parameters:
    model = KMeans(n_clusters=n_clusters)
    model_name = f'KMeans(n_clusters={n_clusters})'
    # Fit model
    model.fit(data_df_train)
    
    # Evaluate model using internal indicators
    cluster_labels_train = model.predict(data_df_train)
    
    # Calculate internal indicators
    if len(np.unique(cluster_labels_train)) > 1:
        SC_score = silhouette_score(data_df_train, cluster_labels_train)
        DB_score = davies_bouldin_score(data_df_train, cluster_labels_train)
        CH_score = calinski_harabasz_score(data_df_train, cluster_labels_train)
    else:
        SC_score = DB_score = CH_score = np.nan
    
    # Calculate external MAE and MAPE on test data
    data_df_test_np = np.array(data_df_test)
    data_df_train_np = np.array(data_df_train)
    best_mae, best_mape = external_score(model, data_df_train_np, data_df_test_np)
    
    # Collect results
    results.append({
        'Model': model_name,
        'Silhouette Score': SC_score,
        'Davies-Bouldin Score': DB_score,
        'Calinski-Harabasz Score': CH_score,
        'External MAE': best_mae,
        'External MAPE': best_mape
    })

# Iterate over AgglomerativeClustering parameters
for n_clusters in agglo_parameters:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    model_name = f'AgglomerativeClustering(n_clusters={n_clusters})'
    # Fit model
    model.fit(data_df_train)
    
    # Since AgglomerativeClustering doesn't have predict method, use labels_
    cluster_labels_train = model.labels_
    
    # Calculate internal indicators
    if len(np.unique(cluster_labels_train)) > 1:
        SC_score = silhouette_score(data_df_train, cluster_labels_train)
        DB_score = davies_bouldin_score(data_df_train, cluster_labels_train)
        CH_score = calinski_harabasz_score(data_df_train, cluster_labels_train)
    else:
        SC_score = DB_score = CH_score = np.nan
    
    # Calculate external MAE and MAPE on test data
    best_mae, best_mape = external_score(model, data_df_train_np, data_df_test_np)
    
    # Collect results
    results.append({
        'Model': model_name,
        'Silhouette Score': SC_score,
        'Davies-Bouldin Score': DB_score,
        'Calinski-Harabasz Score': CH_score,
        'External MAE': best_mae,
        'External MAPE': best_mape
    })

# Iterate over combinations of DBSCAN parameters
for eps in dbscan_eps:
    for min_samples in dbscan_min_samples:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model_name = f'DBSCAN(eps={eps}, min_samples={min_samples})'
        # Fit model
        model.fit(data_df_train)
        
        cluster_labels_train = model.labels_
        
        # Check if there are clusters other than noise
        if len(np.unique(cluster_labels_train)) > 1 and len(set(cluster_labels_train)) > (1 if -1 in cluster_labels_train else 0):
            SC_score = silhouette_score(data_df_train, cluster_labels_train)
            DB_score = davies_bouldin_score(data_df_train, cluster_labels_train)
            CH_score = calinski_harabasz_score(data_df_train, cluster_labels_train)
            
            # Calculate external MAE and MAPE on test data
            best_mae, best_mape = external_score(model, data_df_train_np, data_df_test_np)
        else:
            SC_score = DB_score = CH_score = np.nan
            best_mae = best_mape = np.nan  # Cannot compute external score if no valid clusters
        
        # Collect results
        results.append({
            'Model': model_name,
            'Silhouette Score': SC_score,
            'Davies-Bouldin Score': DB_score,
            'Calinski-Harabasz Score': CH_score,
            'External MAE': best_mae,
            'External MAPE': best_mape
        })

# Iterate over GaussianMixture parameters
for n_components in gmm_parameters:
    model = GaussianMixture(n_components=n_components)
    model_name = f'GaussianMixture(n_components={n_components})'
    # Fit model
    model.fit(data_df_train)
    
    # Predict cluster labels
    cluster_labels_train = model.predict(data_df_train)
    
    # Calculate internal indicators
    if len(np.unique(cluster_labels_train)) > 1:
        SC_score = silhouette_score(data_df_train, cluster_labels_train)
        DB_score = davies_bouldin_score(data_df_train, cluster_labels_train)
        CH_score = calinski_harabasz_score(data_df_train, cluster_labels_train)
    else:
        SC_score = DB_score = CH_score = np.nan
    
    # Calculate external MAE and MAPE on test data
    best_mae, best_mape = external_score(model, data_df_train_np, data_df_test_np)
    
    # Collect results
    results.append({
        'Model': model_name,
        'Silhouette Score': SC_score,
        'Davies-Bouldin Score': DB_score,
        'Calinski-Harabasz Score': CH_score,
        'External MAE': best_mae,
        'External MAPE': best_mape
    })

# Print results
for res in results:
    print(res)
