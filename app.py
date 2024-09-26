from flask import Flask, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from flask import request
app = Flask(__name__)


# route to accept new ratings and update matrix
@app.route('/rate', methods=['POST'])
def add_rating():
    data = request.get_json()

    user_id = data['user_id']
    item_id = data['item_id']
    rating  =  data['rating']
    

    # check if user and item exist in the matrix
    if user_id >= user_item_matrix.shape[0]:
        #if user is new ad them to the matrix 
        new_user_row = pd.Series([0] * user_item_matrix.shape[1], index=user_item_matrix.columns)
        user_item_matrix.loc[user_id] = new_user_row
    
    if item_id not in user_item_matrix.columns:
        # if the item is new add it o the matrix
        user_item_matrix[item_id] =0
    
    #update the useritem matric with the new rating 
    user_item_matrix.at[user_id,item_id] = rating

    #update the knn model with the new user item matrix 
    global user_item_matrix_np
    user_item_matrix_np = user_item_matrix.values
    knn.fit(user_item_matrix_np)

    return jsonify({
        'message': 'rating added successfully',
        'user_id': user_id,
        'item_id': item_id,
        'rating': rating
    }),200

# Load the dataset and preprocess it
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('data/u.data', sep='\t', names=column_names)
df = df.drop('timestamp', axis=1)

# Create user-item matrix
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Convert user-item matrix into a NumPy array
user_item_matrix_np = user_item_matrix.values

# Initialize the KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix_np)

# Train the KNN model for user-based recommendations
user_knn = NearestNeighbors(metric='cosine', algorithm='brute')
user_knn.fit(user_item_matrix_np)

# Train the KNN model for item-based recommendations
item_knn = NearestNeighbors(metric='cosine', algorithm='brute')
item_knn.fit(user_item_matrix.T)  # Transpose the matrix for item-based recommendations


def get_recommendations(user_id):
    # Check if user ID exists in the dataset
    if user_id >= user_item_matrix.shape[0]:
        return {"error": "User ID not found"}
    
    # Find similar users
    distances, indices = knn.kneighbors(user_item_matrix_np[user_id].reshape(1, -1), n_neighbors=6)

    # Get recommended items from similar users (excluding the user itself)
    similar_users = indices.flatten()[1:]  # Skip the first index (the user itself)
    recommended_items = set()

    for similar_user in similar_users:
        # Get items that the similar user has rated (convert Series to NumPy array and find non-zero elements)
        similar_user_items = user_item_matrix.iloc[similar_user].to_numpy().nonzero()[0]
        recommended_items.update(similar_user_items)
        
    # Convert numpy.int64 to native Python int
    return [int(item) for item in recommended_items]

# Sample route to return recommendations for a user
# @app.route('/recommend/<int:user_id>', methods=['GET'])
# def recommend(user_id):
#     recommendations = get_recommendations(user_id)
#     if isinstance(recommendations, dict) and 'error' in recommendations:
#         return jsonify(recommendations), 404
#     return jsonify({
#         'user_id': user_id,
#         'recommendations': recommendations
#     })


    #New function to get item-based recommendations
def get_item_based_recommendations(user_id, num_recommendations=5):
    # Check if user_id exists in the dataset
    if user_id >= user_item_matrix.shape[0]:
        return {"error": "User ID not found"}

    # Get the user's ratings
    user_ratings = user_item_matrix.iloc[user_id]

    # Find items the user has rated highly (e.g., rating 4 or 5)
    highly_rated_items = user_ratings[user_ratings >= 4].index.tolist()

    # Find items similar to the highly rated ones
    similar_items = set()
    for item in highly_rated_items:
        # Get the k nearest neighbors (similar items)
        item_vector = user_item_matrix[item].values.reshape(1, -1)
        distances, indices = item_knn.kneighbors(user_item_matrix.T.loc[item].values.reshape(1, -1), n_neighbors=num_recommendations + 1)

        # Add similar items to the recommendation list
        for idx in indices[0]:
            if idx != item:  # Don't recommend the same item
                similar_items.add(idx)
    return list(similar_items)

#Update the recommendation route to include item_based recommendation 
@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    user_based_recommendations = get_recommendations(user_id)
    item_based_recommendations = get_item_based_recommendations(user_id)

    # Convert numpy.int64 to native int for JSON serialization
    user_based_recommendations = [int(item) for item in user_based_recommendations]
    item_based_recommendations = [int(item) for item in item_based_recommendations]

    if isinstance(user_based_recommendations, dict) and 'error' in user_based_recommendations:
        return jsonify(user_based_recommendations), 404
    
    return jsonify({
        'user_id': user_id,
        'user_based_recommendations': user_based_recommendations,
        'item_based_recommendations': item_based_recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
