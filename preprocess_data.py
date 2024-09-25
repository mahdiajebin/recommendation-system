import pandas as pandas
from sklearn.neighbors import NearestNeighbors
import numpy as np
#load dataset
column_names = ['user_id', 'item_id','rating','timestamp']
df = pandas.read_csv('data/u.data',sep='\t',names=column_names)

#check the first fewq rows of dataset
print(df.head())

#check for missing values
print(df.isnull().sum())

#drop the timestamp column
df=df.drop('timestamp',axis=1)

#create a user-item interaction matrix 
user_item_matrix = df.pivot(index='user_id', columns='item_id',values='rating')

#fill missing values with 0 (unrated items)
user_item_matrix = user_item_matrix.fillna(0)

#inspect the matrix
# print (user_item_matrix.head())

# convert the user-item matrix into a Numpy array
user_item_matrix_np = user_item_matrix.values

#initialize the KNN model
knn = NearestNeighbors(metric = 'cosine', algorithm='brute')
knn.fit(user_item_matrix_np)

#Get recommendations for a specific user 
user_index = 0
distances,indices = knn.kneighbors(user_item_matrix_np[user_index].reshape(1,-1), n_neighbors=6)

#print 
print(f"Recommendations for auser {user_index}:")
for i in range (1,len(distances[0])):
    similar_user_index = indices[0][i]
    print(f"Similar user: {similar_user_index} with distance: {distances[0][1]}")
    