from flask import Flask, request, jsonify
import os
from shl_recommendation import SHLRecommendationEngine

app = Flask(__name__)

# Initialize the recommendation engine using an absolute path for the CSV file.
csv_path = os.path.join(os.path.dirname(__file__), 'shl_final_catalog.csv')
engine = SHLRecommendationEngine(csv_path=csv_path)

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Assessment Recommendation Endpoint (POST)
@app.route('/recommend', methods=['POST'])
def recommend():
    if not engine:
        return jsonify({"error": "Engine initialization failed"}), 500

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        recommendations = engine.get_recommendations(data["query"], top_k=10)
        results = []
        for rec in recommendations.to_dict(orient='records'):
            # Process duration: extract digit(s) and convert to integer (default 0)
            duration_value = rec.get("duration", "")
            duration_int = int(''.join(filter(str.isdigit, duration_value))) if any(c.isdigit() for c in duration_value) else 0
            
            # Safely convert test_type to string and split it into an array
            val = rec.get("test_type", "")
            if val is None:
                val = ""
            test_type_str = str(val)
            test_type_array = [t.strip() for t in test_type_str.split(",") if t.strip()]
            
            results.append({
                "url": rec.get("url"),
                "adaptive_support": rec.get("adaptive_support"),
                "description": rec.get("assessment_name"),
                "duration": duration_int,
                "remote_support": rec.get("remote_testing_support"),
                "test_type": test_type_array
            })
        return jsonify({"recommended_assessments": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render usually provides the PORT env variable; default to 10000 locally.
    app.run(host='0.0.0.0', port=port)
