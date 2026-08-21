import os
import pandas as pd
import threading
import re
from groq import Groq

# Global state
reviews_df = pd.DataFrame()
is_initialized = False
is_initializing = False
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

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
        return []
        
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
        
    if not groq_client:
        return "GROQ_API_KEY is not set in the backend environment. Please configure it to use the AI chat."

    try:
        # 1. Search using our ultra-lightweight memory search
        retrieved_reviews = simple_keyword_search(query, top_k=5)
        
        if not retrieved_reviews:
            context_str = "No specific reviews found matching this exact query."
        else:
            context_str = "\n".join([f"- {rev}" for rev in retrieved_reviews])
        
        # 2. Ask Groq AI to generate the answer
        prompt = f"""You are a Zomato Restaurant AI. User asked: "{query}"
Relevant reviews:
{context_str}
Give a short, helpful answer (max 3-4 sentences). Recommend restaurants from reviews. Be friendly. Do not invent restaurants if the user is just saying a greeting."""

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="groq/compound-mini",
            temperature=0.7,
            max_tokens=250
        )
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"RAG Engine Error: {str(e)}"
        print(error_msg)
        return f"Sorry, backend error: {str(e)}"

def add_review(restaurant_name: str, rating: float, review_text: str):
    global reviews_df
    if reviews_df.empty:
        return
        
    # Create a new DataFrame for the new row to avoid warnings
    new_row = pd.DataFrame([{
        'Restaurant': restaurant_name,
        'Rating': rating,
        'Review': review_text,
        'clean_text': review_text.lower()
    }])
    
    # Concatenate the new row
    reviews_df = pd.concat([reviews_df, new_row], ignore_index=True)
    print(f"RAG Engine: Live review added for {restaurant_name}")
