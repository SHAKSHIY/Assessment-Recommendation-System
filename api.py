#api.py
from flask import Flask, request, jsonify
import os
from shl_recommendation import SHLRecommendationEngine

app = Flask(__name__)

# Initialize the recommendation engine once.
engine = SHLRecommendationEngine(csv_path='shl_final_catalog.csv')

@app.route('/recommend', methods=['GET'])
def recommend():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Please provide a query parameter"}), 400
    recommendations = engine.get_recommendations(query, top_k=10)
    recommendations_dict = recommendations.to_dict(orient='records')
    return jsonify(recommendations_dict)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
