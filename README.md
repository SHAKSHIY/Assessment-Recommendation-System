# SHL Assessment Recommendation Engine
## Overview
This project is an intelligent recommendation system that returns up to 10 relevant SHL assessments based on a natural language query or job description. The system processes a catalog of SHL assessments, generates vector embeddings using a SentenceTransformer model, and computes cosine similarity to rank the assessments.

## Features
Data Processing:

Loads assessment data from a CSV file (shl_final_catalog.csv).

Standardizes column names and combines key fields into a single descriptive text.

Embedding Generation:

Uses the SentenceTransformer (all-MiniLM-L6-v2) to convert the combined text into vector embeddings.

Recommendation Logic:

Computes cosine similarity between the query and each assessment embedding using scikit-learn.

Selects and returns the top-K similar assessments as recommendations.

Interactive Web Interface:

Built with Streamlit to allow users to input queries and view recommendations (displaying assessment name with clickable URL, remote testing support, adaptive/IRT support, duration, test type, and similarity score).

REST API:

A Flask API endpoint (/recommend) provides programmatic access to the recommendation engine by returning JSON-formatted results.

## Installation
Prerequisites

Python 3.10 or 3.11

Git

Setup Steps

Clone the Repository:

git clone https://github.com/yourusername/SHL-Assessment-Recommendation-Engine.git

cd SHL-Assessment-Recommendation-Engine

Create a Virtual Environment:

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
