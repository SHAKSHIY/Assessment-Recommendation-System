import pandas as pd
from shl_recommendation import SHLRecommendationEngine

# === STEP 1: Load the model and SHL catalog ===
print("🚀 Loading SHL Recommendation Engine...")
engine = SHLRecommendationEngine(csv_path="shl_final_catalog.csv")

# === STEP 2: Load the unlabeled test queries ===
print("📘 Loading Test Set...")
test_df = pd.read_excel("Test-Set.xlsx", sheet_name=0)  # adjust sheet if needed
print(f"Loaded {len(test_df)} queries.")

# === STEP 3: Generate top-10 recommendations for each query ===
print("🧠 Generating predictions...")
rows = []

for i, query in enumerate(test_df["Query"], start=1):
    try:
        recs = engine.get_recommendations(query, top_k=10)
        for url in recs["url"]:
            rows.append({"Query": query, "Assessment_url": url})
        print(f"✅ Processed {i}/{len(test_df)} queries")
    except Exception as e:
        print(f"⚠️ Error processing query {i}: {e}")

# === STEP 4: Save the results ===
submission_df = pd.DataFrame(rows)
submission_df.to_csv("submission_predictions.csv", index=False)

print("\n🎯 Done! 'submission_predictions.csv' generated successfully.")
