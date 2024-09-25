from flask import Flask,jsonify


app = Flask(__name__)


#sample route to return dummy recommendation
@app.route('/recommend/<int:user_id>',methods=['GET'])
def recommend(user_id):
    recommendations = {
        'user_id':user_id,
        'recommendations': ['item1','item2','item3']
    }
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)