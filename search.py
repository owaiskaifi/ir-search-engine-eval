import os
import pickle
import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class CISISearchEngine:
    def __init__(self, index_dir='output'):
        self.index_dir = index_dir
        self.load_index()
        
        # For preprocessing
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def load_index(self):
        """Load the indexed data."""
        print("Loading index...")
        
        # Load the vectorizer
        with open(os.path.join(self.index_dir, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load document vectors
        with open(os.path.join(self.index_dir, 'doc_vectors.npz'), 'rb') as f:
            loaded = np.load(f)
            from scipy.sparse import csr_matrix
            self.doc_vectors = csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
        
        # Load document IDs
        with open(os.path.join(self.index_dir, 'doc_ids.json'), 'r') as f:
            self.doc_ids = json.load(f)
        
        # Load document metadata
        with open(os.path.join(self.index_dir, 'documents.json'), 'r') as f:
            self.documents = json.load(f)
        
        print(f"Loaded index with {self.doc_vectors.shape[0]} documents and {self.doc_vectors.shape[1]} terms.")
    
    def preprocess_query(self, query_text):
        """Preprocess the query text using the same steps as for documents."""
        # Convert to lowercase and tokenize
        tokens = word_tokenize(query_text.lower())
        
        # Remove stopwords and non-alphanumeric tokens
        tokens = [t for t in tokens if t not in self.stop_words and t.isalnum()]
        
        # Apply stemming
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def search(self, query_text, top_k=10):
        """Search for documents relevant to the query."""
        # Preprocess the query
        preprocessed_query = self.preprocess_query(query_text)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([preprocessed_query])
        
        # Compute cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get the indices of the top-k most similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Get the document IDs and scores
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = similarities[idx]
            
            # Skip documents with zero similarity
            if score <= 0:
                continue
                
            doc_info = self.documents[doc_id]
            results.append({
                'id': doc_id,
                'title': doc_info['title'],
                'author': doc_info['author'],
                'score': float(score),
                'snippet': doc_info['content'][:200] + '...' if len(doc_info['content']) > 200 else doc_info['content']
            })
        
        return results

def main():
    # Initialize the search engine
    search_engine = CISISearchEngine()
    
    # Interactive search loop
    print("\nCISI Search Engine")
    print("Enter a query or 'quit' to exit")
    
    while True:
        query = input("\nQuery: ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        # Search for relevant documents
        results = search_engine.search(query)
        
        # Display results
        if not results:
            print("No relevant documents found.")
        else:
            print(f"\nFound {len(results)} relevant documents:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [{result['id']}] {result['title']} (Score: {result['score']:.4f})")
                print(f"   Author: {result['author']}")
                print(f"   {result['snippet']}")

if __name__ == "__main__":
    main() 