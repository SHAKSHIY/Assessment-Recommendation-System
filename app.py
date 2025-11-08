# app.py
import sys
import asyncio

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import streamlit as st
from shl_recommendation import SHLRecommendationEngine

# Initialize the recommendation engine (this might take a moment as it loads the model and computes embeddings)
@st.cache_resource(show_spinner=True)
def init_engine():
    return SHLRecommendationEngine(csv_path='shl_final_catalog.csv')

engine = init_engine()

st.title("SHL Assessment Recommendation Engine")
st.write("Enter a natural language query or job description to get assessment recommendations.")

query_input = st.text_input("Enter your query here:")

if st.button("Get Recommendations") and query_input:
    with st.spinner("Computing recommendations..."):
        recs = engine.get_recommendations(query_input, top_k=10)
    st.write("### Top Recommendations:")
    # Display each recommendation in a neat format.
    for idx, row in recs.iterrows():
        st.markdown(f"**Assessment Name:** [{row['assessment_name']}]({row['url']})")
        st.markdown(f"- **Remote Testing Support:** {row['remote_testing_support']}")
        st.markdown(f"- **Adaptive/IRT Support:** {row['adaptive_support']}")
        st.markdown(f"- **Duration:** {row['duration']}")
        st.markdown(f"- **Test Type:** {row['test_type']}")
        st.markdown(f"- **Similarity Score:** {row['similarity']:.2f}")
        st.markdown("---")