from flask import Flask, request, jsonify
import os
import threading
from shl_recommendation import SHLRecommendationEngine

app = Flask(__name__)

# Global variable for the engine
engine = None

def load_engine():
    """
    Load the recommendation engine in the background.
    This may take time, but it won't block the app from binding to a port.
    """
    global engine
    csv_path = os.path.join(os.path.dirname(__file__), 'shl_final_catalog.csv')
    engine = SHLRecommendationEngine(csv_path=csv_path)

# Start loading the engine in a separate thread immediately.
threading.Thread(target=load_engine).start()

# Health Check Endpoint (GET)
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Assessment Recommendation Endpoint (POST)
@app.route('/recommend', methods=['POST'])
def recommend():
    # If the engine has not been loaded yet, return a Service Unavailable error.
    if engine is None:
        return jsonify({"error": "Engine not loaded yet, please try again later."}), 503

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        recommendations = engine.get_recommendations(data["query"], top_k=10)
        results = []
        for rec in recommendations.to_dict(orient='records'):
            # Process duration: convert to string then extract digits.
            duration_value = str(rec.get("duration", ""))
            duration_int = int(''.join(filter(str.isdigit, duration_value))) if any(c.isdigit() for c in duration_value) else 0

            # Process test_type: convert to string safely (handle NaN, float, etc.), then split.
            test_type_value = rec.get("test_type", "")
            if not isinstance(test_type_value, str):
                # Use pandas isna check if needed; otherwise simply convert.
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
        # Optional: print the traceback to logs for debugging.
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # When running locally, bind to PORT from environment (default 10000).
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
