# SHL Assessment Recommendation Engine
![image](https://github.com/user-attachments/assets/584c585c-ae5e-4fbf-bc47-f63fe4bb73ed)

## Overview
This project is an intelligent recommendation system that returns up to 10 relevant SHL assessments based on a natural language query or job description. The system processes a catalog of SHL assessments, generates vector embeddings using a SentenceTransformer model, and computes cosine similarity to rank the assessments.

- Problem: Hiring managers often struggle to select the right SHL assessments for specific roles. Keyword-based filtering is slow and imprecise.
- Solution: A semantic recommendation engine powered by SBERT embeddings and cosine similarity that returns the top K relevant assessments based on a natural‑language query or job description.

  This repository includes:

1. Preprocessing & Embeddings: Scripts to preprocess the SHL catalog and precompute feature embeddings.

2. Flask API: Two endpoints:

   GET /health — simple health check.

   GET /recommend?query=... — returns JSON with up to 10 recommended assessments.

3. Streamlit App: A user-friendly web interface to type queries and browse recommendations.

## Features & Architecture

1. Data & Preprocessing

- Data Source: shl_final_catalog.csv containing columns:

Assessment Name, URL, Remote Testing (Yes/No), Adaptive/IRT (Yes/No), Duration (minutes), Test Type.

- Combined Features: We concatenate all columns into a single combined_features string for each row.

2. Embeddings

- Model: sentence-transformers/all-MiniLM-L6-v2.

- Precompute: Run precompute_embeddings.py once to encode all catalog entries and save as data/embeddings.pkl.

- Cache: On startup, the engine loads the precomputed embeddings for faster initialization.

3. Recommendation Logic

- Query Encoding: Encode the user's natural-language query with the same SBERT model.

- Similarity: Compute cosine similarity between the query embedding and catalog embeddings.

- Top‑K: Sort similarities, take the top K (default 10), and return corresponding assessments.

4. Flask API

- Health Check

GET /health

Response: {"status":"healthy"}

- Recommendation

GET /recommend?query=Your+text+here

Response: [ { url, adaptive_support, description, duration, remote_support, test_type }, … ]

5. Streamlit App

- Initialize cached engine on first run.

- Text input for query, button to trigger recommendations.

- Display each recommendation with hyperlinked name and attributes.

## Installation
Prerequisites

Python 3.10 or 3.11

Git

Setup Steps

- Clone the Repository:

git clone https://github.com/SHAKSHIY/SHL-Assessment-Recommendation-Engine.git

cd SHL-Assessment-Recommendation-Engine

- Create a Virtual Environment:

py -3.11 -m venv venv

Activate the environment:

Windows:

venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

Example requirements.txt:

pandas

numpy

sentence-transformers

scikit-learn

streamlit

flask

## Usage
Running the Web App

Start the Streamlit app:

py -3.11 -m streamlit run app.py

The app will open in your browser at http://localhost:8501.

Running the API

Start the Flask API:

py -3.11 -m python api.py

Access the API endpoint at:

http://localhost:5000/recommend?query=your+query+here

## Deployment

Streamlit Web App:

Deploy on Streamlit Community Cloud by linking your GitHub repository.

Flask API:

Deploy on platforms like Heroku or Render to obtain a publicly accessible API endpoint.

Tools & Libraries

Python 3.11

Pandas & NumPy for data processing

SentenceTransformer, Transformers, Accelerate for embedding generation

scikit-learn for cosine similarity computation

Streamlit for the web interface

Flask for the REST API

## Challenges & Resolutions
Library Compatibility:
Resolved issues such as the init_empty_weights error by installing compatible versions of transformers, accelerate, and sentence-transformers.

## Environment Management:
Leveraged virtual environments and ensured Python 3.10/3.11 was used for stability.
