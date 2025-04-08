from flask import Flask, request, jsonify
import os
from shl_recommendation import SHLRecommendationEngine

app = Flask(__name__)

# Initialize the recommendation engine with the absolute CSV path
csv_path = os.path.join(os.path.dirname(__file__), 'shl_final_catalog.csv')
engine = SHLRecommendationEngine(csv_path=csv_path)

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Assessment Recommendation Endpoint using POST
@app.route('/recommend', methods=['POST'])
def recommend():
    if not engine:
        return jsonify({"error": "Engine initialization failed"}), 500

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        recommendations = engine.get_recommendations(data["query"], top_k=10)
        # Convert the recommendations to our required schema
        results = []
        for rec in recommendations.to_dict(orient='records'):
            # You can modify how to extract 'description'
            # Here we use the 'assessment_name' as a placeholder for description.
            # You can include a more detailed description if available.
            # For duration, extract numbers and convert to integer.
            duration_value = rec.get("duration", "")
            # Extract digit(s) if present, otherwise use 0
            duration_int = int(''.join(filter(str.isdigit, duration_value))) if any(c.isdigit() for c in duration_value) else 0
            
            # Split test type values by comma and trim white spaces
            test_type_array = [t.strip() for t in rec.get("test_type", "").split(",") if t.strip()]
            
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
    # Use the port provided by Render or default to 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
