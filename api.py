from flask import Flask, request, jsonify
import os
from shl_recommendation import SHLRecommendationEngine

app = Flask(__name__)

# Load the engine before starting the server
csv_path = os.path.join(os.path.dirname(__file__), 'shl_final_catalog.csv')
print("🚀 Loading recommendation engine...")
engine = SHLRecommendationEngine(csv_path='shl_final_catalog.csv', model_name='paraphrase-MiniLM-L3-v2')
print("✅ Engine loaded successfully.")

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Recommendation Endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        recommendations = engine.get_recommendations(data["query"], top_k=10)
        results = []

        for rec in recommendations.to_dict(orient='records'):
            # Handle duration
            duration_value = str(rec.get("duration", ""))
            duration_int = int(''.join(filter(str.isdigit, duration_value))) if any(c.isdigit() for c in duration_value) else 0

            # Handle test_type
            test_type_value = rec.get("test_type", "")
            if not isinstance(test_type_value, str):
                try:
                    from pandas import isna
                    if isna(test_type_value):
                        test_type_value = ""
                    else:
                        test_type_value = str(test_type_value)
                except ImportError:
                    test_type_value = str(test_type_value)

            test_type_array = [t.strip() for t in test_type_value.split(",") if t.strip()]

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
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Run locally on port 10000
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port)
