from flask import Flask, request, jsonify
import os
from shl_recommendation import SHLRecommendationEngine

app = Flask(__name__)

# Initialize engine with absolute path
try:
    csv_path = os.path.join(os.path.dirname(__file__), 'shl_final_catalog.csv')
    engine = SHLRecommendationEngine(csv_path=csv_path)
except Exception as e:
    print(f"ERROR: Engine failed to initialize: {e}")
    engine = None

@app.route('/recommend', methods=['GET'])
def recommend():
    if not engine:
        return jsonify({"error": "Engine initialization failed"}), 500
    
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    try:
        recommendations = engine.get_recommendations(query, top_k=10)
        return jsonify(recommendations.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render uses 10000 by default
    app.run(host='0.0.0.0', port=port)
