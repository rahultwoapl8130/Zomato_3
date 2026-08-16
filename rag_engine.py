import os
import chromadb
import google.generativeai as genai
import pandas as pd
import threading

# Global state
collection = None
is_initialized = False
is_initializing = False
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def init_rag_async(df_reviews: pd.DataFrame):
    global collection, is_initialized, is_initializing
    
    if is_initialized or is_initializing:
        return
        
    is_initializing = True
    
    try:
        # Initialize ChromaDB client (local persistent storage)
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(name="zomato_reviews")
        
        # Check if DB is already populated
        if collection.count() == 0 and not df_reviews.empty:
            print("Populating Vector DB. This may take a few minutes...")
            
            documents = []
            metadatas = []
            ids = []
            
            # Subsetting for demo to avoid huge startup times. Production would do all.
            df_subset = df_reviews.tail(2000).reset_index(drop=True)
            
            for idx, row in df_subset.iterrows():
                if pd.isna(row.get('Review')):
                    continue
                
                restaurant = str(row.get('Restaurant', 'Unknown'))
                rating = str(row.get('Rating', '0'))
                review_text = str(row.get('Review', ''))
                
                doc = f"Restaurant: {restaurant}. Rating: {rating} stars. Review: {review_text}"
                
                documents.append(doc)
                metadatas.append({
                    "restaurant": restaurant,
                    "rating": float(rating) if rating.replace('.', '', 1).isdigit() else 0
                })
                ids.append(f"rev_{idx}")
                
                if len(documents) >= 500:
                    collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    documents, metadatas, ids = [], [], []
            
            if documents:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print("Vector DB Populated successfully!")
            
        is_initialized = True
    except Exception as e:
        print(f"Failed to initialize Vector DB: {e}")
    finally:
        is_initializing = False

def init_rag(df_reviews: pd.DataFrame):
    # Run in background to avoid blocking FastAPI startup
    thread = threading.Thread(target=init_rag_async, args=(df_reviews,))
    thread.daemon = True
    thread.start()

def query_rag(query: str):
    if not is_initialized or collection is None:
        return "System is still initializing the Vector Database. Please try again in a moment."
        
    if not GEMINI_API_KEY:
        return "GEMINI_API_KEY is not set in the backend environment. Please configure it to use the AI chat."

    try:
        results = collection.query(
            query_texts=[query],
            n_results=10
        )
        
        retrieved_reviews = results['documents'][0] if results['documents'] else []
        context_str = "\n".join([f"- {rev}" for rev in retrieved_reviews])
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
You are an expert Restaurant AI Assistant. 
A user has asked: "{query}"

Here are the most relevant customer reviews from our database:
{context_str}

Based ONLY on the provided reviews, answer the user's question. 
Be concise, helpful, and friendly. Do not hallucinate.
"""
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"RAG Error: {e}")
        return f"Sorry, I encountered an error while processing your request. Please try again."
