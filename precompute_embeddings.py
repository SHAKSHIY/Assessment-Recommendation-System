import pandas as pd
import pickle
from shl_recommendation import SHLRecommendationEngine as RecommendationEngine


# Load the catalog
df = pd.read_csv("shl_final_catalog.csv")

# Create the engine and compute embeddings
engine = RecommendationEngine()

# Save the engine to a file
with open("data/embeddings.pkl", "wb") as f:
    pickle.dump(engine, f)

print("✅ Embeddings saved to data/embeddings.pkl")
