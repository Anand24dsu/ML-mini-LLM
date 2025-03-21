import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
import faiss
import json

# Load and preprocess dataset
file_path = './hotel_bookings.csv'

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = df.ffill()  # Forward fill missing values
        # Create a description column for semantic search
        df['description'] = df.apply(lambda row: f"Hotel: {row['hotel']}, Year: {row['arrival_date_year']}, Month: {row['arrival_date_month']}, Lead Time: {row['lead_time']} days, Canceled: {row['is_canceled']}", axis=1)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Generate Analytics
def compute_analytics(df):
    try:
        revenue_trend = df.groupby('arrival_date_year')['adr'].sum()
        cancellation_rate = (df['is_canceled'].sum() / len(df)) * 100
        geo_distribution = df['hotel'].value_counts()
        lead_time_distribution = df['lead_time'].describe()
        return {
            'revenue_trend': revenue_trend.to_dict(),
            'cancellation_rate': cancellation_rate,
            'geo_distribution': geo_distribution.to_dict(),
            'lead_time_distribution': lead_time_distribution.to_dict()
        }
    except Exception as e:
        return {"error": f"Error computing analytics: {e}"}

# Setup FAISS for RAG
class VectorDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.data = []

    def add_data(self, texts):
        vectors = self.model.encode(texts)
        self.index.add(np.array(vectors))
        self.data.extend(texts)

    def search(self, query, top_k=1):
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector), top_k)
        return [self.data[idx] for idx in indices[0]] if len(indices[0]) > 0 else []

# Flask Setup
app = Flask(__name__)
db = VectorDB()

data = load_data(file_path)
if not data.empty:
    db.add_data(data['description'].tolist())

@app.route("/analytics", methods=["GET"])
def get_analytics():
    if data.empty:
        return jsonify({"error": "Data not loaded properly"}), 500
    return jsonify(compute_analytics(data))

@app.route("/ask", methods=["POST"])
def ask_question():
    if not db.data:
        return jsonify({"error": "Vector database not initialized"}), 500
    request_data = request.get_json()
    if not request_data or "question" not in request_data:
        return jsonify({"error": "Missing 'question' field in request"}), 400
    question = request_data.get("question", "")
    response = db.search(question)
    return jsonify({"answer": response})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Hotel Booking Data API"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7600, debug=True)
