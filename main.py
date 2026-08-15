from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import json
import random

class PredictionRequest(BaseModel):
    text: str
    restaurant: str = None


app = FastAPI(title="Zomato Restaurant AI Backend")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, replace "*" with the Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

try:
    df_meta = pd.read_csv(os.path.join(DATA_DIR, "Zomato Restaurant names and Metadata.csv"))
    df_reviews = pd.read_csv(os.path.join(DATA_DIR, "Zomato Restaurant reviews.csv"))
except Exception as e:
    print(f"Error loading CSV files: {e}")
    df_meta = pd.DataFrame()
    df_reviews = pd.DataFrame()

# Helper for Sentiment (Mocking ML model behavior until actual model is integrated)
def get_sentiment(rating):
    try:
        r = float(rating)
        if r >= 4: return 'Positive'
        if r <= 2.5: return 'Negative'
        return 'Neutral'
    except:
        return 'Neutral'

@app.get("/")
def read_root():
    return {"message": "Welcome to Zomato AI Backend API"}

@app.get("/api/restaurants")
def get_restaurants():
    if df_meta.empty:
        return []
        
    restaurants = []
    
    # Pre-calculate sentiment scores for all restaurants
    restaurant_sentiments = {}
    if not df_reviews.empty:
        for name, group in df_reviews.groupby('Restaurant'):
            positive_count = 0
            total_count = len(group)
            for _, r_row in group.iterrows():
                rating = float(r_row.get('Rating', 0)) if pd.notna(r_row.get('Rating')) and str(r_row.get('Rating')).replace('.','',1).isdigit() else 0
                sentiment = get_sentiment(rating)
                if sentiment == 'Positive':
                    positive_count += 1
            
            score = int((positive_count / total_count) * 100) if total_count > 0 else 0
            restaurant_sentiments[name] = score

    for idx, row in df_meta.iterrows():
        cuisines = [c.strip() for c in str(row.get('Cuisines', '')).split(',')] if pd.notna(row.get('Cuisines')) else []
        cost_str = str(row.get('Cost', '500')).replace(',', '')
        cost = int(cost_str) if cost_str.isdigit() else 500
        name = row.get('Name', '')
        link = str(row.get('Links', ''))
        
        # Use calculated sentiment score or fallback to a default (e.g., 50)
        score = restaurant_sentiments.get(name, 50)

        restaurants.append({
            "id": f"r{idx+1}",
            "name": name,
            "location": "Hyderabad",
            "rating": round(random.uniform(3.0, 5.0), 1),
            "costForTwo": cost,
            "cuisines": cuisines,
            "image": f"https://loremflickr.com/400/300/food,dish?lock={idx+1}",
            "sentimentScore": score,
            "deliveryTime": "30-45 min",
            "link": link
        })
    return restaurants

@app.get("/api/restaurants/{restaurant_id}")
def get_restaurant_detail(restaurant_id: str):
    if df_meta.empty:
        return {"error": "No data available"}
        
    # Find restaurant by id (e.g. 'r1')
    try:
        idx = int(restaurant_id.replace('r', '')) - 1
        if idx < 0 or idx >= len(df_meta):
            return {"error": "Restaurant not found"}
            
        row = df_meta.iloc[idx]
        name = row.get('Name', '')
        
        # Get reviews
        restaurant_reviews = []
        if not df_reviews.empty:
            rev_df = df_reviews[df_reviews['Restaurant'] == name]
            for i, r_row in rev_df.iterrows():
                restaurant_reviews.append({
                    "id": f"{name}-rev-{i}",
                    "customerName": r_row.get('Reviewer', 'Anonymous'),
                    "rating": float(r_row.get('Rating', 0)) if pd.notna(r_row.get('Rating')) and str(r_row.get('Rating')).replace('.','',1).isdigit() else 0,
                    "text": str(r_row.get('Review', '')),
                    "date": str(r_row.get('Time', 'Recently')),
                    "sentiment": get_sentiment(r_row.get('Rating', 0))
                })
        
        cuisines = [c.strip() for c in str(row.get('Cuisines', '')).split(',')] if pd.notna(row.get('Cuisines')) else []
        cost_str = str(row.get('Cost', '500')).replace(',', '')
        cost = int(cost_str) if cost_str.isdigit() else 500
        link = str(row.get('Links', ''))
        
        # Calculate real sentiment for detail page
        positive_count = sum(1 for rev in restaurant_reviews if rev["sentiment"] == 'Positive')
        total_count = len(restaurant_reviews)
        score = int((positive_count / total_count) * 100) if total_count > 0 else 50
        
        return {
            "id": restaurant_id,
            "name": name,
            "location": "Hyderabad",
            "rating": round(random.uniform(3.0, 5.0), 1),
            "costForTwo": cost,
            "cuisines": cuisines,
            "image": f"https://loremflickr.com/400/300/food,dish?lock={idx+1}",
            "sentimentScore": score,
            "reviews": restaurant_reviews,
            "link": link
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/dashboard")
def get_dashboard():
    restaurants = get_restaurants()
    sorted_restaurants = sorted(restaurants, key=lambda x: x['sentimentScore'], reverse=True)
    top5 = sorted_restaurants[:5]
    bottom5 = sorted_restaurants[-5:]
    bottom5.reverse()
    
    # Mock some reviews for live feed
    all_reviews = []
    if not df_reviews.empty:
        # Get first 50 valid reviews
        for i, r_row in df_reviews.head(50).iterrows():
            all_reviews.append({
                "id": f"live-rev-{i}",
                "customerName": r_row.get('Reviewer', 'Anonymous'),
                "rating": float(r_row.get('Rating', 0)) if pd.notna(r_row.get('Rating')) and str(r_row.get('Rating')).replace('.','',1).isdigit() else 0,
                "text": str(r_row.get('Review', '')),
                "date": str(r_row.get('Time', 'Recently')),
                "sentiment": get_sentiment(r_row.get('Rating', 0))
            })
            
    sentimentData = [
      {"month": 'Jan', "positive": 6500, "negative": 1200, "neutral": 3400},
      {"month": 'Feb', "positive": 6800, "negative": 1100, "neutral": 3200},
      {"month": 'Mar', "positive": 7100, "negative": 1300, "neutral": 3100},
      {"month": 'Apr', "positive": 8500, "negative": 1000, "neutral": 2800},
      {"month": 'May', "positive": 9200, "negative": 800, "neutral": 2400},
      {"month": 'Jun', "positive": 10500, "negative": 700, "neutral": 2000},
    ]
    
    cuisine_counts = {}
    for r in restaurants:
        for c in r['cuisines']:
            cuisine_counts[c] = cuisine_counts.get(c, 0) + 1
            
    cuisineData = [{"name": k, "value": v} for k, v in sorted(cuisine_counts.items(), key=lambda item: item[1], reverse=True)[:5]]
    
    return {
        "top5": top5,
        "bottom5": bottom5,
        "reviews": all_reviews,
        "sentimentData": sentimentData,
        "cuisineData": cuisineData
    }

@app.post("/api/predict")
def predict_rating(request: PredictionRequest):
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Review text is required")
        
    # Since .pkl models are not available in the deployment environment,
    # we simulate the AI prediction based on heuristics and keywords for now
    # to provide a clean API contract for the frontend without breaking deployment.
    
    text_lower = request.text.lower()
    
    # Simple heuristic-based keyword scoring
    positive_words = ['good', 'great', 'excellent', 'amazing', 'best', 'delicious', 'tasty', 'awesome', 'nice', 'love', 'perfect']
    negative_words = ['bad', 'terrible', 'worst', 'awful', 'poor', 'disgusting', 'bland', 'cold', 'late', 'rude', 'hate']
    
    pos_score = sum(1 for word in positive_words if word in text_lower)
    neg_score = sum(1 for word in negative_words if word in text_lower)
    
    if pos_score > neg_score:
        sentiment = 'Positive'
        rating = random.uniform(4.0, 5.0)
    elif neg_score > pos_score:
        sentiment = 'Negative'
        rating = random.uniform(1.0, 2.5)
    else:
        sentiment = 'Neutral'
        rating = random.uniform(2.6, 3.9)
        
    confidence = random.randint(75, 98)
    
    # Simulated SHAP explainability
    factors = []
    if sentiment == 'Positive':
        factors = [{"feature": "Food Quality", "impact": "+"}, {"feature": "Taste", "impact": "+"}]
    elif sentiment == 'Negative':
        factors = [{"feature": "Service Speed", "impact": "-"}, {"feature": "Taste", "impact": "-"}]
        
    return {
        "prediction": round(rating, 1),
        "sentiment": sentiment,
        "confidence": confidence,
        "model": "Simulated (LightGBM/BERT unavailable)",
        "explainability": factors
    }
