import pandas as pd
from shl_recommendation import SHLRecommendationEngine

# === Load engine ===
engine = SHLRecommendationEngine(csv_path="shl_final_catalog.csv")

# === Load train dataset ===
train_df = pd.read_excel("Train-set.xlsx")
print(f"Loaded {len(train_df)} labeled queries for evaluation.")

hits = 0
total = 0

for i, row in train_df.iterrows():
    query = str(row["Query"])
    true_url = str(row["Assessment_url"]).strip().lower()
    
    recs = engine.get_recommendations(query, top_k=10)
    predicted_urls = [str(u).strip().lower() for u in recs["url"].tolist()]
    
    if any(true_url in p for p in predicted_urls):
        hits += 1
    total += 1
    
    print(f"[{i+1}/{len(train_df)}] {'✅' if true_url in predicted_urls else '❌'}")

recall_at_10 = hits / total if total > 0 else 0
print(f"\n📊 Recall@10 = {recall_at_10:.2%}")
