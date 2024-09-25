from flask import Flask, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

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
@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    recommendations = get_recommendations(user_id)
    if isinstance(recommendations, dict) and 'error' in recommendations:
        return jsonify(recommendations), 404
    return jsonify({
        'user_id': user_id,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)
