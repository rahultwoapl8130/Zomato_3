import os
import pandas as pd
import threading
import re
import numpy as np
from groq import Groq

# Global state
reviews_df = pd.DataFrame()
is_initialized = False
is_initializing = False
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Advanced RAG components
embedding_model = None
faiss_index = None
bm25_model = None
corpus_texts = []
corpus_metadata = []

def init_rag_async(df_reviews: pd.DataFrame):
    global reviews_df, is_initialized, is_initializing
    global embedding_model, faiss_index, bm25_model, corpus_texts, corpus_metadata
    
    if is_initialized or is_initializing:
        return
        
    is_initializing = True
    
    try:
        import gc
        if not df_reviews.empty:
            print("Initializing Advanced RAG DB (Hybrid + FAISS)...")
            
            # Subsetting to 800 to avoid Render's 512MB RAM OOM crash
            df_subset = df_reviews.dropna(subset=['Review']).tail(800).reset_index(drop=True)
            reviews_df = df_subset
            
            del df_reviews
            gc.collect()
            
            # 1. Load Sentence Transformer (Lazy load to save memory initially)
            from sentence_transformers import SentenceTransformer
            import faiss
            from rank_bm25 import BM25Okapi
            
            print("Loading Embedding Model...")
            # all-MiniLM-L6-v2 is ~80MB, fast and lightweight
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 2. Prepare Corpus
            print("Preparing Corpus...")
            for idx, row in reviews_df.iterrows():
                restaurant = str(row.get('Restaurant', 'Unknown'))
                rating = str(row.get('Rating', '0'))
                review_text = str(row.get('Review', ''))
                
                # We embed a combination of restaurant name and review for better context
                combined_text = f"Restaurant: {restaurant}. Rating: {rating} stars. Review: {review_text}"
                corpus_texts.append(combined_text)
                corpus_metadata.append({
                    "restaurant": restaurant,
                    "rating": rating,
                    "review": review_text
                })
            
            # 3. Create FAISS Vector Index
            print("Building FAISS Index...")
            embeddings = embedding_model.encode(corpus_texts, convert_to_numpy=True)
            dimension = embeddings.shape[1]
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss_index.add(embeddings)
            
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
    Performs FAISS Vector Search + BM25 Keyword Search, 
    and combines them using Reciprocal Rank Fusion (RRF).
    """
    if not is_initialized or not faiss_index or not bm25_model:
        return []
        
    # 1. Semantic Search (FAISS)
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    distances, faiss_indices = faiss_index.search(query_vector, top_k * 2)
    
    faiss_results = {}
    for rank, idx in enumerate(faiss_indices[0]):
        if idx < len(corpus_texts):
            faiss_results[idx] = rank + 1
            
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
    
    all_indices = set(list(faiss_results.keys()) + list(bm25_results.keys()))
    
    for idx in all_indices:
        score = 0.0
        if idx in faiss_results:
            score += 1.0 / (k + faiss_results[idx])
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

def check_needs_retrieval(query: str) -> bool:
    """Self-RAG Check: Does this query need database retrieval?"""
    prompt = f"""Analyze the user's input and determine if it requires searching a restaurant database.
If it is a greeting, casual chat, or general knowledge question, answer NO.
If it is asking for food recommendations, restaurant details, or reviews, answer YES.

User input: "{query}"

Reply strictly with YES or NO."""

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="groq/compound-mini",
            temperature=0.0,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    except:
        return True # Default to True if error

def generate_hypothetical_answer(query: str) -> str:
    """HyDE: Generate a hypothetical review to improve semantic search"""
    prompt = f"""You are a customer writing a Zomato review. Write a short, fake review (2-3 sentences) that perfectly answers the user's query.
Do not include real restaurant names, just use descriptive words related to the vibe, food, and experience.

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

def query_rag(query: str):
    if not is_initialized:
        return "System is still initializing the Advanced RAG Database (Loading FAISS and Transformers). Please try again in a moment."
        
    if not groq_client:
        return "GROQ_API_KEY is not set."

    try:
        # 1. Self-RAG: Check if we need to search
        needs_search = check_needs_retrieval(query)
        
        if not needs_search:
            # Answer directly without context
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": f"You are a friendly Zomato Foodie Assistant. Answer the user casually. User: {query}"}],
                model="groq/compound-mini",
                temperature=0.7,
                max_tokens=250
            )
            return response.choices[0].message.content
            
        # 2. HyDE: Generate hypothetical document
        hypothetical_doc = generate_hypothetical_answer(query)
        
        # Combine original query and hypothetical doc for a super-powered search
        search_query = f"{query} {hypothetical_doc}"
        
        # 3. Hybrid Search (FAISS + BM25)
        retrieved_reviews = hybrid_search(search_query, top_k=5)
        
        if not retrieved_reviews:
            context_str = "No specific reviews found matching this query."
        else:
            context_str = "\n".join([f"- {rev}" for rev in retrieved_reviews])
        
        # 4. Final Generation
        prompt = f"""You are a friendly and enthusiastic AI Foodie Assistant! You help hungry users find amazing places to eat using real Zomato reviews.

YOUR BEHAVIOR:
- Read the provided reviews carefully and highlight *why* people liked the restaurant.
- Keep it short and sweet. Use food emojis to make the conversation lively.
- If the reviews don't seem relevant, say 'Oops! My data doesn't have a good match for that right now.'

Relevant reviews:
{context_str}

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
    global reviews_df, corpus_texts, corpus_metadata, faiss_index, bm25_model
    if not is_initialized or not embedding_model:
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
    
    # Update FAISS
    new_vector = embedding_model.encode([combined_text], convert_to_numpy=True)
    faiss_index.add(new_vector)
    
    # Update BM25 (Requires rebuilding index, but it's fast for 3000 docs)
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
    bm25_model = BM25Okapi(tokenized_corpus)
    
    print(f"Advanced RAG: Live review added and indexed for {restaurant_name}")
