from flask import Flask, request, jsonify
import os
from shl_recommendation import SHLRecommendationEngine
import traceback
import pandas as pd  # to use pd.isna for better NaN detection

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
            # Process duration: convert duration field to string and extract digits
            duration_value = str(rec.get("duration", ""))
            duration_int = int(''.join(filter(str.isdigit, duration_value))) if any(c.isdigit() for c in duration_value) else 0

            # Process test_type: convert to string safely, handling NaN values from pandas
            test_type_value = rec.get("test_type", "")
            if not isinstance(test_type_value, str):
                # If it's not a string, check if it is NaN (NaN is not equal to itself)
                if pd.isna(test_type_value):
                    test_type_value = ""
                else:
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
        # Return the results in the required JSON format
        return jsonify({"recommended_assessments": results}), 200

    except Exception as e:
        tb = traceback.format_exc()
        print("Error processing /recommend request:\n", tb)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use PORT from environment (Render sets it) or default to 10000 for local testing.
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
