# shl_recommendation.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: import only when using Streamlit
try:
    import streamlit as st
    from huggingface_hub import login
    # Try to safely get the Hugging Face API token from st.secrets
    huggingface_token = None
    if hasattr(st, "secrets"):
        huggingface_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
    if huggingface_token:
        login(token=huggingface_token)
except (ImportError, ModuleNotFoundError):
    # Not running in Streamlit or streamlit module not available
    pass
except Exception as e:
    print("Streamlit secrets not found or error accessing them, skipping Hugging Face login.")

class SHLRecommendationEngine:
    def __init__(self, csv_path='shl_final_catalog.csv', model_name='all-MiniLM-L6-v2'):
        self.csv_path = csv_path
        self.model_name = model_name
        self.catalog = self.load_and_preprocess_data()
        self.model = SentenceTransformer(self.model_name)
        self.embeddings = self.compute_embeddings()

    def load_and_preprocess_data(self):
        """
        Load the SHL catalog CSV and preprocess it.
        Assumes the CSV has the following columns:
         - Assessment Name
         - URL
         - Remote Testing
         - Adaptive/IRT
         - Duration
         - Test Type
        This function renames them to lowercase keys.
        """
        catalog = pd.read_csv(self.csv_path)
        
        # Rename columns to standard names
        catalog.rename(columns={
            'Assessment Name': 'assessment_name',
            'URL': 'url',
            'Remote Testing': 'remote_testing_support',
            'Adaptive/IRT': 'adaptive_support',
            'Duration': 'duration',
            'Test Type': 'test_type'
        }, inplace=True)
        
        # Create a combined text field for embedding generation.
        catalog['combined_features'] = (
            catalog['assessment_name'].astype(str) + " " +
            catalog['remote_testing_support'].astype(str) + " " +
            catalog['adaptive_support'].astype(str) + " " +
            catalog['duration'].astype(str) + " " +
            catalog['test_type'].astype(str)
        )
        return catalog

    def compute_embeddings(self):
        """
        Compute embeddings for all catalog entries using the combined features.
        """
        texts = self.catalog['combined_features'].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def get_recommendations(self, query, top_k=10):
        """
        Given a natural language query, compute its embedding and return the top_k recommendations.
        Returns a DataFrame with recommended assessments and their details.
        """
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        recommendations = self.catalog.iloc[top_indices].copy()
        recommendations['similarity'] = similarities[top_indices]
        return recommendations

# Example usage (for testing purposes):
if __name__ == '__main__':
    engine = SHLRecommendationEngine(csv_path='shl_final_catalog.csv')
    test_query = "I am hiring for a role that requires quick decision making and strong communication skills."
    recs = engine.get_recommendations(test_query, top_k=5)
    # Note: now columns are renamed, so we refer to them with lowercase names.
    print(recs[['assessment_name', 'url', 'remote_testing_support', 'adaptive_support', 'duration', 'test_type', 'similarity']])
