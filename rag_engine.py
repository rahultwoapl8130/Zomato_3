import os
import google.generativeai as genai
import pandas as pd
import threading
import re

# Global state
reviews_df = pd.DataFrame()
is_initialized = False
is_initializing = False
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def init_rag_async(df_reviews: pd.DataFrame):
    global reviews_df, is_initialized, is_initializing
    
    if is_initialized or is_initializing:
        return
        
    is_initializing = True
    
    try:
        if not df_reviews.empty:
            print("Initializing Lightweight Search DB (No ChromaDB)...")
            
            # Subsetting for demo to avoid huge memory/startup times. 
            # We keep only valid reviews and store them in memory.
            df_subset = df_reviews.dropna(subset=['Review']).tail(3000).reset_index(drop=True)
            
            # Basic preprocessing for fast search
            df_subset['clean_text'] = df_subset['Review'].astype(str).str.lower()
            
            reviews_df = df_subset
            print(f"Lightweight Search DB Populated with {len(reviews_df)} records successfully!")
            
        is_initialized = True
    except Exception as e:
        print(f"Failed to initialize Lightweight Search DB: {e}")
    finally:
        is_initializing = False

def init_rag(df_reviews: pd.DataFrame):
    # Run in background to avoid blocking FastAPI startup
    thread = threading.Thread(target=init_rag_async, args=(df_reviews,))
    thread.daemon = True
    thread.start()

def simple_keyword_search(query: str, top_k: int = 10):
    """
    Very lightweight keyword matching instead of heavy vector embeddings.
    Perfect for 512MB RAM environments like Render Free Tier.
    """
    if reviews_df.empty:
        return []
        
    # Extract words from query (ignore small words)
    words = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 2]
    
    if not words:
        # Fallback: just return random highly rated reviews
        return reviews_df.head(top_k).to_dict('records')
        
    # Score each review based on how many query words it contains
    def score_review(text):
        return sum(1 for w in words if w in text)
        
    # Calculate scores (vectorized approach would be better but this is fine for small subset)
    scores = reviews_df['clean_text'].apply(score_review)
    
    # Get indices of top matches (must have at least 1 matching word)
    top_indices = scores[scores > 0].nlargest(top_k).index
    
    if len(top_indices) == 0:
        return []
        
    results = []
    for idx in top_indices:
        row = reviews_df.iloc[idx]
        restaurant = str(row.get('Restaurant', 'Unknown'))
        rating = str(row.get('Rating', '0'))
        review_text = str(row.get('Review', ''))
        results.append(f"Restaurant: {restaurant}. Rating: {rating} stars. Review: {review_text}")
        
    return results

def query_rag(query: str):
    if not is_initialized or reviews_df.empty:
        return "System is still initializing the Restaurant database. Please try again in a moment."
        
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY is not set in the backend environment. Please configure it to use the AI chat."

    try:
        # 1. Search using our ultra-lightweight memory search
        retrieved_reviews = simple_keyword_search(query, top_k=15)
        
        if not retrieved_reviews:
            context_str = "No specific reviews found matching this exact query."
        else:
            context_str = "\n".join([f"- {rev}" for rev in retrieved_reviews])
        
        # 2. Ask Gemini AI to generate the answer
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = f"""
You are an expert Restaurant AI Assistant for Zomato. 
A user has asked: "{query}"

Here are the most relevant customer reviews from our database:
{context_str}

Based ONLY on the provided reviews, answer the user's question. 
If they ask for recommendations, suggest the restaurants mentioned positively in the reviews.
Be concise, helpful, and friendly. Do not hallucinate.
"""
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        error_msg = f"RAG Engine Error: {str(e)}"
        print(error_msg)
        return f"Sorry, backend error: {str(e)}"
