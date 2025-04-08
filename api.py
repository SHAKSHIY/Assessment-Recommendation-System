from flask import Flask, request, jsonify
import os
from shl_recommendation import SHLRecommendationEngine

app = Flask(__name__)

# Compute the absolute CSV path
csv_path = os.path.join(os.path.dirname(__file__), 'shl_final_catalog.csv')

# Initialize the recommendation engine
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
            # Process duration: ensure duration is handled as string first, then extract digits.
            duration_value = str(rec.get("duration", ""))
            duration_int = int(''.join(filter(str.isdigit, duration_value))) if any(c.isdigit() for c in duration_value) else 0
            
            # Process test_type: convert the value to string and split based on comma
            test_type_value = rec.get("test_type", "")
            test_type_str = str(test_type_value) if test_type_value is not None else ""
            test_type_array = [t.strip() for t in test_type_str.split(",") if t.strip()]
            
            results.append({
                "url": rec.get("url"),
                "adaptive_support": rec.get("adaptive_support"),
                "description": rec.get("assessment_name"),
                "duration": duration_int,
                "remote_support": rec.get("remote_testing_support"),
                "test_type": test_type_array
            })
        # Return in the required format
        return jsonify({"recommended_assessments": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use the PORT environment variable if available (Render will set this), otherwise default to 10000.
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
