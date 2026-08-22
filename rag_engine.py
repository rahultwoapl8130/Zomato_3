import os
import pandas as pd
import threading
import numpy as np
from groq import Groq
import gc
from typing import List, Dict

# Global state
reviews_df = pd.DataFrame()
is_initialized = False
is_initializing = False
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Advanced Lightweight RAG components
tfidf_vectorizer = None
tfidf_matrix = None
bm25_model = None
corpus_texts = []
corpus_metadata = []

def init_rag_async(df_reviews: pd.DataFrame):
    global reviews_df, is_initialized, is_initializing
    global tfidf_vectorizer, tfidf_matrix, bm25_model, corpus_texts, corpus_metadata
    
    if is_initialized or is_initializing:
        return
        
    is_initializing = True
    
    try:
        if not df_reviews.empty:
            print("Initializing Advanced RAG DB (Hybrid + TF-IDF)...")
            
            # Using 1500 reviews - TF-IDF is memory efficient
            df_subset = df_reviews.dropna(subset=['Review']).tail(1500).reset_index(drop=True)
            reviews_df = df_subset
            
            del df_reviews
            gc.collect()
            
            # 1. Load ML models (scikit-learn is lightweight)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from rank_bm25 import BM25Okapi
            
            # 2. Prepare Corpus
            print("Preparing Corpus...")
            for idx, row in reviews_df.iterrows():
                restaurant = str(row.get('Restaurant', 'Unknown'))
                rating = str(row.get('Rating', '0'))
                review_text = str(row.get('Review', ''))
                
                # Combine for better context
                combined_text = f"Restaurant: {restaurant}. Rating: {rating} stars. Review: {review_text}"
                corpus_texts.append(combined_text)
                corpus_metadata.append({
                    "restaurant": restaurant,
                    "rating": rating,
                    "review": review_text
                })
            
            # 3. Create TF-IDF Vector Index (Lightweight Semantic Approximation)
            print("Building TF-IDF Index...")
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)
            
            # 4. Create BM25 Keyword Index
            print("Building BM25 Index...")
            tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
            bm25_model = BM25Okapi(tokenized_corpus)
            
            print(f"Advanced RAG DB Populated with {len(reviews_df)} records successfully!")
            
        is_initialized = True
    except Exception as e:
        print(f"Failed to initialize Advanced RAG DB: {e}")
    finally:
        is_initializing = False

def init_rag(df_reviews: pd.DataFrame):
    # Run in background to avoid blocking FastAPI startup
    thread = threading.Thread(target=init_rag_async, args=(df_reviews,))
    thread.daemon = True
    thread.start()

def hybrid_search(query: str, top_k: int = 5):
    """
    Performs TF-IDF Vector Search + BM25 Keyword Search, 
    and combines them using Reciprocal Rank Fusion (RRF).
    """
    if not is_initialized or tfidf_matrix is None or not bm25_model:
        return []
        
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 1. Semantic-like Search (TF-IDF)
    query_vec = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Get top_k*2 indices sorted by score descending
    tfidf_indices = cosine_similarities.argsort()[::-1][:top_k * 2]
    
    tfidf_results = {}
    for rank, idx in enumerate(tfidf_indices):
        if cosine_similarities[idx] > 0:
            tfidf_results[idx] = rank + 1
            
    # 2. Keyword Search (BM25)
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25_model.get_scores(tokenized_query)
    # Get top_k*2 indices sorted by score descending
    bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
    
    bm25_results = {}
    for rank, idx in enumerate(bm25_indices):
        if bm25_scores[idx] > 0: # Only if it actually matched something
            bm25_results[idx] = rank + 1
            
    # 3. Reciprocal Rank Fusion (RRF)
    # RRF Score = 1 / (k + rank) where k is a constant (usually 60)
    k = 60
    rrf_scores = {}
    
    all_indices = set(list(tfidf_results.keys()) + list(bm25_results.keys()))
    
    for idx in all_indices:
        score = 0.0
        if idx in tfidf_results:
            score += 1.0 / (k + tfidf_results[idx])
        if idx in bm25_results:
            score += 1.0 / (k + bm25_results[idx])
        rrf_scores[idx] = score
        
    # Sort by RRF score descending
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # Return top_k docs
    results = []
    for idx in sorted_indices[:top_k]:
        results.append(corpus_texts[idx])
        
    return results

def check_needs_retrieval(query: str, history_str: str) -> str:
    """Self-RAG Check: Determine action (NO_SEARCH, DB_SEARCH, WEB_SEARCH)"""
    prompt = f"""Analyze the user's input and conversation history to determine the next action.
- If it is a simple greeting or general knowledge off-topic question, answer NO_SEARCH.
- If the user asks for restaurant recommendations, reviews, or details about food in India, answer DB_SEARCH.
- If the user specifically asks for web search, very recent news, or a completely unknown global restaurant, answer WEB_SEARCH.

Conversation History:
{history_str}

User input: "{query}"

Reply strictly with ONE word: NO_SEARCH, DB_SEARCH, or WEB_SEARCH."""

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="groq/compound-mini",
            temperature=0.0,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().upper()
        if "WEB_SEARCH" in answer: return "WEB_SEARCH"
        if "NO_SEARCH" in answer: return "NO_SEARCH"
        return "DB_SEARCH"
    except:
        return "DB_SEARCH"

def generate_hypothetical_answer(query: str, history_str: str) -> str:
    """HyDE: Generate a hypothetical review to improve semantic search"""
    prompt = f"""You are a customer writing a Zomato review. Write a short, fake review (2-3 sentences) that perfectly answers the user's query given their chat history.
Do not include real restaurant names, just use descriptive words related to the vibe, food, and experience.

History:
{history_str}

User query: "{query}"

Fake Review:"""

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="groq/compound-mini",
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except:
        return ""

def do_web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(f"restaurant {query}", max_results=3)
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search failed: {str(e)}"

def format_history(history: List[Dict[str, str]]) -> str:
    if not history: return "None"
    return "\n".join([f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}" for msg in history[-4:]]) # Keep last 4 for context

def query_rag(query: str, history: List[Dict[str, str]] = [], preference: str = "All"):
    if not is_initialized:
        return "System is still initializing the Advanced RAG Database. Please try again in a moment."
        
    if not groq_client:
        return "GROQ_API_KEY is not set."

    try:
        history_str = format_history(history)
        
        # 1. Self-RAG: Check action
        action = check_needs_retrieval(query, history_str)
        
        pref_rule = ""
        if preference.lower() == "veg":
            pref_rule = "CRITICAL: The user is a Strict Vegetarian. You MUST ONLY recommend Veg food. Ignore any meat reviews."
        elif preference.lower() == "non-veg":
            pref_rule = "The user loves Non-Veg. Highlight meat dishes if applicable."
            
        system_base = f"""You are a friendly Zomato Foodie Assistant.
Your ONLY job is to help users find great food, recommend restaurants, and chat about dining.
Always reply in a friendly **Hinglish** tone (a mix of Hindi and English written in English script). Example: 'Bhai, yeh try kar!' or 'Main ekdum theek hoon, aaj kya khane ka mann hai?'.
{pref_rule}"""

        if action == "NO_SEARCH":
            # Answer directly without context but with strict topic guardrails
            guardrail_prompt = f"""{system_base}
- If the user sends a simple greeting, reply warmly in Hinglish and ask what they want to eat.
- If the user asks a general knowledge, political, scientific, or off-topic question, DO NOT answer it. Politely refuse in Hinglish.

Chat History:
{history_str}

User: "{query}"
"""
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": guardrail_prompt}],
                model="groq/compound-mini",
                temperature=0.7,
                max_tokens=250
            )
            return response.choices[0].message.content
            
        if action == "WEB_SEARCH":
            web_results = do_web_search(query)
            prompt = f"""{system_base}
I performed a live web search for the user's query since it wasn't in our local DB.
Web Results:
{web_results}

Chat History:
{history_str}

User: "{query}"
"""
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="groq/compound-mini",
                temperature=0.7,
                max_tokens=250
            )
            return response.choices[0].message.content

        # action == "DB_SEARCH"
        # 2. HyDE: Generate hypothetical document
        hypothetical_doc = generate_hypothetical_answer(query, history_str)
        
        # Combine original query and hypothetical doc for a super-powered search
        # Also append the last user message from history for context
        last_user_msg = ""
        for m in reversed(history):
            if m.get('role') == 'user':
                last_user_msg = m.get('content')
                break
                
        search_query = f"{last_user_msg} {query} {hypothetical_doc}"
        
        # 3. Hybrid Search (TF-IDF + BM25)
        retrieved_reviews = hybrid_search(search_query, top_k=5)
        
        if not retrieved_reviews:
            context_str = "No specific reviews found matching this query."
        else:
            context_str = "\n".join([f"- {rev}" for rev in retrieved_reviews])
        
        # 4. Final Generation
        prompt = f"""{system_base}

YOUR BEHAVIOR:
- Read the provided reviews carefully and highlight *why* people liked the restaurant.
- Keep it short and sweet. Use food emojis to make the conversation lively.
- Reply in friendly **Hinglish**.
- If the reviews don't seem relevant, say 'Oops! Mere paas iska exact match nahi mila.' in Hinglish.

Relevant reviews:
{context_str}

Chat History:
{history_str}

User asked: "{query}"
"""

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
    global reviews_df, corpus_texts, corpus_metadata, tfidf_matrix, tfidf_vectorizer, bm25_model
    if not is_initialized or not tfidf_vectorizer:
        return
        
    # Update dataframe
    new_row = pd.DataFrame([{
        'Restaurant': restaurant_name,
        'Rating': rating,
        'Review': review_text
    }])
    reviews_df = pd.concat([reviews_df, new_row], ignore_index=True)
    
    # Format new document
    combined_text = f"Restaurant: {restaurant_name}. Rating: {rating} stars. Review: {review_text}"
    corpus_texts.append(combined_text)
    corpus_metadata.append({
        "restaurant": restaurant_name,
        "rating": rating,
        "review": review_text
    })
    
    # Update TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)
    
    # Update BM25
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
    bm25_model = BM25Okapi(tokenized_corpus)
    
    print(f"Advanced RAG: Live review added and indexed for {restaurant_name}")
