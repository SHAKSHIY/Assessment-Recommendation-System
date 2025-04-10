import os
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
from sklearn.preprocessing import normalize


# Get Hugging Face token from environment variables
huggingface_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if huggingface_token:
    login(token=huggingface_token)
else:
    print("⚠️ Hugging Face token not found in environment variables.")

class SHLRecommendationEngine:
    def __init__(self, csv_path='shl_final_catalog.csv', model_name='paraphrase-MiniLM-L3-v2'):
        self.csv_path = csv_path
        self.model_name = model_name
        self.catalog = self.load_and_preprocess_data()
        self.model = SentenceTransformer(self.model_name)
        
        embeddings_path = "data/embeddings.pkl"
        if os.path.exists(embeddings_path):
            with open(embeddings_path, "rb") as f:
                self.embeddings = pickle.load(f)
            print("✅ Loaded precomputed embeddings.")
        else:
            print("⚠️ Precomputed embeddings not found. Computing now...")
            self.embeddings = self.compute_embeddings()
            os.makedirs("data", exist_ok=True)
            with open(embeddings_path, "wb") as f:
                pickle.dump(self.embeddings, f)
            print("✅ Computed and saved new embeddings.")

    def load_and_preprocess_data(self):
        catalog = pd.read_csv(self.csv_path)
        catalog.rename(columns={
            'Assessment Name': 'assessment_name',
            'URL': 'url',
            'Remote Testing': 'remote_testing_support',
            'Adaptive/IRT': 'adaptive_support',
            'Duration': 'duration',
            'Test Type': 'test_type'
        }, inplace=True)

        catalog['combined_features'] = (
            catalog['assessment_name'].astype(str) + " " +
            catalog['remote_testing_support'].astype(str) + " " +
            catalog['adaptive_support'].astype(str) + " " +
            catalog['duration'].astype(str) + " " +
            catalog['test_type'].astype(str)
        )
        return catalog

    def compute_embeddings(self):
        texts = self.catalog['combined_features'].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def get_recommendations(self, query, top_k=10):
    # 1) Encode the query and ensure it's a float array
        raw_query_emb = self.model.encode([query])  # shape (1, embedding_dim)
        query_embedding = np.array(raw_query_emb, dtype=np.float32)

    # 2) Convert your precomputed (or computed) embeddings to float arrays too
        embeddings_float = np.array(self.embeddings, dtype=np.float32)

    # 3) Now compute cosine similarity using numeric arrays
        similarities = cosine_similarity(query_embedding, embeddings_float)[0]

    # 4) Find top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

    # 5) Copy recommended rows
        recommendations = self.catalog.iloc[top_indices].copy()
        recommendations['similarity'] = similarities[top_indices]
        return recommendations

if __name__ == '__main__':
    engine = SHLRecommendationEngine(csv_path='shl_final_catalog.csv')
    test_query = "I am hiring for a role that requires quick decision making and strong communication skills."
    recs = engine.get_recommendations(test_query, top_k=5)
    print(recs[['assessment_name', 'url', 'remote_testing_support', 'adaptive_support', 'duration', 'test_type', 'similarity']])
