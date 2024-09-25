import pandas as pd  # changed the alias to 'pd' for consistency
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load dataset
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('data/u.data', sep='\t', names=column_names)

# Check the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop the timestamp column
df = df.drop('timestamp', axis=1)

# Create a user-item interaction matrix
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating')

# Fill missing values with 0 (unrated items)
user_item_matrix = user_item_matrix.fillna(0)

# Convert the user-item matrix into a Numpy array
user_item_matrix_np = user_item_matrix.values

# Initialize the KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix_np)

# Get recommendations for a specific user 
user_index = 0
distances, indices = knn.kneighbors(user_item_matrix_np[user_index].reshape(1, -1), n_neighbors=6)

# Print recommendations
print(f"Recommendations for user {user_index}:")
for i in range(1, len(distances[0])):  # start from 1 to skip the user itself
    similar_user_index = indices[0][i]
    print(f"Similar user: {similar_user_index} with distance: {distances[0][i]}")
