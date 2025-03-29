import os
import pickle
import json
import numpy as np
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class RankingMethods:
    def __init__(self, index_dir='output'):
        self.index_dir = index_dir
        self.load_index()
        
        # For preprocessing
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Term frequency in entire corpus - for BM25
        self.term_freq_in_corpus = {}
        self.compute_corpus_term_frequencies()
        
        # Calculate document lengths - for BM25
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.compute_doc_lengths()
    
    def load_index(self):
        """Load the indexed data."""
        print("Loading index...")
        
        # Load the vectorizer
        with open(os.path.join(self.index_dir, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load document vectors
        with open(os.path.join(self.index_dir, 'doc_vectors.npz'), 'rb') as f:
            loaded = np.load(f)
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
        
        return tokens
    
    def compute_corpus_term_frequencies(self):
        """Compute the term frequencies across the entire corpus for BM25"""
        vocabulary = self.vectorizer.get_feature_names_out()
        
        # Count term occurrences across all documents
        for term_idx, term in enumerate(vocabulary):
            # Get non-zero entries for this term (column) in the TF-IDF matrix
            col = self.doc_vectors.getcol(term_idx)
            doc_count = col.nnz  # Number of documents containing this term
            self.term_freq_in_corpus[term] = doc_count
    
    def compute_doc_lengths(self):
        """Compute document lengths for BM25"""
        total_length = 0
        num_docs = len(self.doc_ids)
        
        for doc_id in self.doc_ids:
            text = self.documents[doc_id]['full_text']
            tokens = word_tokenize(text.lower())
            filtered_tokens = [t for t in tokens if t not in self.stop_words and t.isalnum()]
            doc_length = len(filtered_tokens)
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length
        
        self.avg_doc_length = total_length / num_docs if num_docs > 0 else 0
    
    def rank_tfidf_cosine(self, query_text, top_k=10):
        """Rank using TF-IDF with cosine similarity (original method)"""
        # Preprocess the query
        preprocessed_query = ' '.join(self.preprocess_query(query_text))
        
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
                'snippet': doc_info['content'][:200] + '...' if len(doc_info['content']) > 200 else doc_info['content'],
                'ranking_method': 'TF-IDF Cosine'
            })
        
        return results
    
    def rank_bm25(self, query_text, top_k=10, k1=1.5, b=0.75):
        """Rank using BM25 (Okapi BM25) formula
        
        BM25 is a probabilistic ranking function that extends the TF-IDF model with
        document length normalization and term saturation.
        
        Parameters:
        - k1: Term frequency saturation parameter (typically between 1.2 and 2.0)
        - b: Document length normalization parameter (typically 0.75)
        """
        query_terms = self.preprocess_query(query_text)
        
        if not query_terms:
            return []
        
        vocabulary = self.vectorizer.get_feature_names_out()
        vocab_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
        
        # Get query term indices that exist in our vocabulary
        query_term_indices = [vocab_to_idx[term] for term in query_terms if term in vocab_to_idx]
        
        # Calculate BM25 scores for each document
        scores = {}
        N = len(self.doc_ids)  # Total number of documents
        
        # For each document
        for idx, doc_id in enumerate(self.doc_ids):
            score = 0
            doc_length = self.doc_lengths[doc_id]
            
            # For each term in the query
            for term_idx in query_term_indices:
                if term_idx >= self.doc_vectors.shape[1]:
                    continue
                    
                # Get term frequency in document
                term_freq = self.doc_vectors[idx, term_idx]
                
                # Skip if term not in document
                if term_freq == 0:
                    continue
                
                # Get document frequency of term
                term = vocabulary[term_idx]
                doc_freq = self.term_freq_in_corpus.get(term, 0)
                
                # Calculate IDF (Inverse Document Frequency)
                if doc_freq > 0:
                    idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
                else:
                    idf = 0
                
                # BM25 term-document score
                numerator = term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                term_score = idf * (numerator / denominator if denominator != 0 else 0)
                
                score += term_score
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort documents by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for doc_id, score in ranked_docs:
            doc_info = self.documents[doc_id]
            results.append({
                'id': doc_id,
                'title': doc_info['title'],
                'author': doc_info['author'],
                'score': float(score),
                'snippet': doc_info['content'][:200] + '...' if len(doc_info['content']) > 200 else doc_info['content'],
                'ranking_method': 'BM25'
            })
        
        return results
    
    def rank_boolean_extended(self, query_text, top_k=10):
        """Rank using Extended Boolean Model with TF-IDF weights
        
        This is an extended Boolean model that combines Boolean logic with
        vector space weighting to provide ranked results.
        """
        query_terms = self.preprocess_query(query_text)
        
        if not query_terms:
            return []
        
        vocabulary = self.vectorizer.get_feature_names_out()
        vocab_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
        
        # Get query term indices that exist in our vocabulary
        query_term_indices = [vocab_to_idx[term] for term in query_terms if term in vocab_to_idx]
        
        # Calculate Extended Boolean scores for each document
        scores = {}
        
        for idx, doc_id in enumerate(self.doc_ids):
            # Calculate sum of squares of weights
            sum_squares = 0
            missing_terms_count = 0
            
            # For each term in the query
            for term_idx in query_term_indices:
                if term_idx >= self.doc_vectors.shape[1]:
                    missing_terms_count += 1
                    continue
                    
                # Get term weight in document
                term_weight = self.doc_vectors[idx, term_idx]
                
                if term_weight == 0:
                    missing_terms_count += 1
                else:
                    sum_squares += term_weight ** 2
            
            # Compute the extended Boolean score (p-norm with p=2)
            # This is similar to the vector space model but respects Boolean-like constraints
            if len(query_term_indices) > 0:
                # Normalize by number of query terms
                normalized_sum = sum_squares / len(query_term_indices)
                
                # Penalize for missing terms
                penalty = 1.0 - (missing_terms_count / len(query_term_indices)) ** 2
                
                # Final score: combine normalized sum with penalty
                score = math.sqrt(normalized_sum) * penalty
                
                if score > 0:
                    scores[doc_id] = score
        
        # Sort documents by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for doc_id, score in ranked_docs:
            doc_info = self.documents[doc_id]
            results.append({
                'id': doc_id,
                'title': doc_info['title'],
                'author': doc_info['author'],
                'score': float(score),
                'snippet': doc_info['content'][:200] + '...' if len(doc_info['content']) > 200 else doc_info['content'],
                'ranking_method': 'Extended Boolean'
            })
        
        return results
    
    def rank_proximity(self, query_text, top_k=10, window_size=10):
        """Rank using Term Proximity
        
        This ranking method takes into account the proximity of query terms
        in the document. Documents with query terms appearing closer together
        receive higher scores.
        """
        query_terms = self.preprocess_query(query_text)
        
        if len(query_terms) <= 1:
            # Fall back to TF-IDF for single-term queries
            results = self.rank_tfidf_cosine(query_text, top_k)
            for result in results:
                result['ranking_method'] = 'Proximity (fallback to TF-IDF)'
            return results
        
        # Calculate proximity scores
        scores = {}
        
        for doc_id in self.doc_ids:
            # Get document text and preprocess
            doc_text = self.documents[doc_id]['full_text'].lower()
            tokens = word_tokenize(doc_text)
            tokens = [self.stemmer.stem(t) for t in tokens if t not in self.stop_words and t.isalnum()]
            
            # Find positions of query terms in document
            term_positions = {}
            for i, token in enumerate(tokens):
                if token in query_terms:
                    if token not in term_positions:
                        term_positions[token] = []
                    term_positions[token].append(i)
            
            # Skip if not all query terms are in the document
            if len(term_positions) < len(query_terms):
                continue
            
            # Calculate minimum span containing all query terms
            min_span = float('inf')
            total_occurrences = 1  # To avoid division by zero
            
            # For each position of the first query term
            for positions in term_positions.values():
                total_occurrences += len(positions)
            
            # Find minimum span for each combination of term positions
            # This is a simplified algorithm that calculates spans between consecutive terms
            min_spans = []
            
            # Get all positions sorted
            all_positions = []
            for term, positions in term_positions.items():
                for pos in positions:
                    all_positions.append((pos, term))
            
            all_positions.sort()
            
            # Find spans containing all query terms
            current_terms = set()
            start_idx = 0
            
            for i in range(len(all_positions)):
                pos, term = all_positions[i]
                current_terms.add(term)
                
                # If we have all terms, calculate span
                if len(current_terms) == len(query_terms):
                    span = pos - all_positions[start_idx][0] + 1
                    min_spans.append(span)
                    
                    # Move start pointer forward until we lose a term
                    removed_term = all_positions[start_idx][1]
                    start_idx += 1
                    current_terms.remove(removed_term)
                    
                    # Add the term back if it appears again in the window
                    for j in range(start_idx, i):
                        if all_positions[j][1] == removed_term:
                            current_terms.add(removed_term)
                            break
            
            # If we found spans, use the minimum one for scoring
            if min_spans:
                min_span = min(min_spans)
                
                # Apply proximity scoring formula
                # Formula: (1 / min_span) * tf_idf_score * (total_occurrences / window_size)
                
                # Get TF-IDF score as base
                query_vector = self.vectorizer.transform([' '.join(query_terms)])
                doc_idx = self.doc_ids.index(doc_id)
                tfidf_score = cosine_similarity(query_vector, self.doc_vectors[doc_idx:doc_idx+1])[0][0]
                
                # Calculate proximity boost
                proximity_boost = 1.0 / min_span if min_span > 0 else 1.0
                occurrence_boost = min(1.0, total_occurrences / window_size)
                
                # Final score
                score = tfidf_score * (1 + proximity_boost * occurrence_boost)
                
                if score > 0:
                    scores[doc_id] = score
            
        # Sort documents by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for doc_id, score in ranked_docs:
            doc_info = self.documents[doc_id]
            results.append({
                'id': doc_id,
                'title': doc_info['title'],
                'author': doc_info['author'],
                'score': float(score),
                'snippet': doc_info['content'][:200] + '...' if len(doc_info['content']) > 200 else doc_info['content'],
                'ranking_method': 'Term Proximity'
            })
        
        return results

def main():
    # Initialize the ranking methods
    ranker = RankingMethods()
    
    # Interactive search loop
    print("\nEnhanced CISI Search Engine with Multiple Ranking Methods")
    print("Enter a query or 'quit' to exit")
    
    while True:
        query = input("\nQuery: ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        print("\nRetrieving and ranking documents...")
        
        # Rank using different methods
        results_tfidf = ranker.rank_tfidf_cosine(query)
        results_bm25 = ranker.rank_bm25(query)
        results_boolean = ranker.rank_boolean_extended(query)
        
        # Display results side by side for comparison
        print("\n===== Comparison of Ranking Methods =====")
        
        # Find how many results we got across all methods
        all_results = []
        all_results.extend(results_tfidf[:3] if results_tfidf else [])
        all_results.extend(results_bm25[:3] if results_bm25 else [])
        all_results.extend(results_boolean[:3] if results_boolean else [])
        
        if not all_results:
            print("No relevant documents found.")
            continue
        
        # Display top results from each method
        for method_name, results in [
            ("1. TF-IDF with Cosine Similarity", results_tfidf),
            ("2. BM25", results_bm25),
            ("3. Extended Boolean Model", results_boolean)
        ]:
            print(f"\n--- {method_name} ---")
            
            if not results:
                print("  No relevant documents found with this method.")
                continue
                
            for i, result in enumerate(results[:3], 1):  # Show top 3 for each method
                print(f"  {i}. [{result['id']}] {result['title']} (Score: {result['score']:.4f})")
                print(f"     Author: {result['author']}")
                print(f"     {result['snippet'][:100]}..." if len(result['snippet']) > 100 else result['snippet'])
        
        # Ask which method the user prefers
        method_choice = input("\nWhich ranking method do you prefer? (1, 2, or 3): ")
        
        # Display full results for chosen method
        chosen_results = None
        if method_choice == '1':
            chosen_results = results_tfidf
            print("\nShowing full results for TF-IDF with Cosine Similarity:")
        elif method_choice == '2':
            chosen_results = results_bm25
            print("\nShowing full results for BM25:")
        elif method_choice == '3':
            chosen_results = results_boolean
            print("\nShowing full results for Extended Boolean Model:")
        
        if chosen_results:
            for i, result in enumerate(chosen_results, 1):
                print(f"\n{i}. [{result['id']}] {result['title']} (Score: {result['score']:.4f})")
                print(f"   Author: {result['author']}")
                print(f"   {result['snippet']}")

if __name__ == "__main__":
    main() 