import numpy as np
import math
from scipy.sparse import csr_matrix
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class RankingFormulas:
    """
    Implementation of three standard information retrieval ranking formulas:
    1. BM25 (Okapi BM25) - A probabilistic ranking function
    2. TF-IDF - Term Frequency-Inverse Document Frequency statistical measure
    3. Cosine Similarity - Vector space model approach
    """
    
    def __init__(self, documents):
        """
        Initialize the ranking formulas with a collection of documents.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Preprocess documents
        self.doc_tokens = [self.preprocess(doc) for doc in documents]
        
        # Calculate document frequencies and corpus statistics
        self.calculate_document_statistics()
        
    def preprocess(self, text):
        """
        Preprocess text by tokenizing, removing stopwords, and stemming.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphanumeric tokens
        tokens = [t for t in tokens if t not in self.stopwords and t.isalnum()]
        
        # Apply stemming
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def calculate_document_statistics(self):
        """
        Calculate various document statistics needed for ranking formulas:
        - Document frequencies
        - Document lengths
        - Vocabulary
        - TF-IDF vectors
        """
        self.N = len(self.documents)  # Number of documents
        
        # Calculate document lengths
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
        # Build vocabulary and calculate document frequencies
        self.vocabulary = set()
        self.df = {}  # Document frequency
        
        for tokens in self.doc_tokens:
            # Add unique terms in this document to vocabulary
            unique_terms = set(tokens)
            self.vocabulary.update(unique_terms)
            
            # Increment document frequency for each term
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1
        
        # Create term-to-index mapping
        self.term_to_idx = {term: idx for idx, term in enumerate(sorted(self.vocabulary))}
        
        # Calculate term frequencies for each document
        self.tf = []  # Term frequencies
        for tokens in self.doc_tokens:
            term_counts = Counter(tokens)
            self.tf.append(term_counts)
            
        # Compute TF-IDF matrix
        self.create_tfidf_matrix()
        
    def create_tfidf_matrix(self):
        """
        Create a TF-IDF matrix for all documents.
        """
        vocab_size = len(self.vocabulary)
        self.tfidf_matrix = np.zeros((self.N, vocab_size))
        
        for doc_idx, term_counts in enumerate(self.tf):
            for term, count in term_counts.items():
                if term in self.term_to_idx:
                    term_idx = self.term_to_idx[term]
                    
                    # Calculate TF (term frequency)
                    tf = count / self.doc_lengths[doc_idx] if self.doc_lengths[doc_idx] > 0 else 0
                    
                    # Calculate IDF (inverse document frequency)
                    idf = math.log(self.N / (self.df[term] + 1)) + 1
                    
                    # Calculate TF-IDF
                    self.tfidf_matrix[doc_idx, term_idx] = tf * idf
        
        # Normalize TF-IDF vectors (for cosine similarity)
        row_norms = np.sqrt(np.sum(self.tfidf_matrix ** 2, axis=1))
        for i in range(self.N):
            if row_norms[i] > 0:
                self.tfidf_matrix[i] = self.tfidf_matrix[i] / row_norms[i]
        
    #-------------------------------------------------------------------------
    # FORMULA 1: TF-IDF (Term Frequency-Inverse Document Frequency)
    #-------------------------------------------------------------------------
    def tfidf_score(self, query, doc_idx):
        """
        Calculate TF-IDF score between a query and a document.
        
        This is a statistical measure used to evaluate how important a word is to a document
        in a collection or corpus:
        - TF (term frequency) measures how frequently a term occurs in a document
        - IDF (inverse document frequency) measures how important a term is across all documents
        
        Formula: score(q, d) = ∑(tf(t, d) * idf(t))
        where:
        - tf(t, d) is the term frequency of term t in document d
        - idf(t) is the inverse document frequency of term t
        
        Args:
            query: Query string
            doc_idx: Document index
            
        Returns:
            TF-IDF score
        """
        # Preprocess query
        query_tokens = self.preprocess(query)
        
        # Get document tokens
        doc_tokens = self.doc_tokens[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        # Calculate score
        score = 0
        
        for term in query_tokens:
            if term in self.df and term in doc_tokens:
                # Calculate TF (term frequency in document)
                tf = self.tf[doc_idx].get(term, 0) / doc_length if doc_length > 0 else 0
                
                # Calculate IDF (inverse document frequency)
                idf = math.log(self.N / (self.df[term] + 1)) + 1
                
                # Add to score
                score += tf * idf
        
        return score
    
    #-------------------------------------------------------------------------
    # FORMULA 2: Cosine Similarity
    #-------------------------------------------------------------------------
    def cosine_similarity(self, query, doc_idx):
        """
        Calculate cosine similarity between a query and a document.
        
        Cosine similarity measures the cosine of the angle between two non-zero vectors.
        It's used in the vector space model for information retrieval to determine
        how similar a document is to a query.
        
        Formula: similarity(q, d) = (q · d) / (||q|| * ||d||)
        where:
        - q · d is the dot product of the query and document vectors
        - ||q|| and ||d|| are the magnitudes (Euclidean norms) of the vectors
        
        Args:
            query: Query string
            doc_idx: Document index
            
        Returns:
            Cosine similarity score (between 0 and 1)
        """
        # Preprocess query
        query_tokens = self.preprocess(query)
        
        # Create query vector
        query_vector = np.zeros(len(self.vocabulary))
        
        # Compute query term frequencies
        query_tf = Counter(query_tokens)
        
        # Populate query vector with TF-IDF values
        for term, count in query_tf.items():
            if term in self.term_to_idx:
                term_idx = self.term_to_idx[term]
                
                # Calculate TF (term frequency)
                tf = count / len(query_tokens) if query_tokens else 0
                
                # Calculate IDF (inverse document frequency)
                idf = math.log(self.N / (self.df.get(term, 0) + 1)) + 1
                
                # Calculate TF-IDF
                query_vector[term_idx] = tf * idf
        
        # Normalize query vector
        query_norm = np.sqrt(np.sum(query_vector ** 2))
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Get document vector
        doc_vector = self.tfidf_matrix[doc_idx]
        
        # Calculate cosine similarity (dot product of normalized vectors)
        similarity = np.dot(query_vector, doc_vector)
        
        return similarity
    
    #-------------------------------------------------------------------------
    # FORMULA 3: BM25 (Okapi BM25)
    #-------------------------------------------------------------------------
    def bm25_score(self, query, doc_idx, k1=1.5, b=0.75):
        """
        Calculate BM25 score between a query and a document.
        
        BM25 (Best Matching 25) is a probabilistic ranking function used to
        estimate the relevance of documents to a given search query. It's an
        extension of TF-IDF that adds term frequency saturation and document
        length normalization.
        
        Formula: score(q, d) = ∑ IDF(t) * (f(t, d) * (k1 + 1)) / (f(t, d) + k1 * (1 - b + b * |d|/avgdl))
        where:
        - f(t, d) is the term frequency of term t in document d
        - |d| is the length of document d
        - avgdl is the average document length in the corpus
        - k1 and b are free parameters:
          - k1 controls term frequency saturation (typically between 1.2 and 2.0)
          - b controls document length normalization (typically 0.75)
        
        Args:
            query: Query string
            doc_idx: Document index
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
            
        Returns:
            BM25 score
        """
        # Preprocess query
        query_tokens = self.preprocess(query)
        
        # Get document length and term frequencies
        doc_length = self.doc_lengths[doc_idx]
        
        # Calculate score
        score = 0
        
        for term in query_tokens:
            if term in self.df:
                # Calculate document frequency (df) and term frequency (tf)
                df = self.df[term]
                tf = self.tf[doc_idx].get(term, 0)
                
                # Skip if term not in document
                if tf == 0:
                    continue
                
                # Calculate IDF component
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                
                # Calculate numerator (tf component with saturation)
                numerator = tf * (k1 + 1)
                
                # Calculate denominator (length normalization)
                denominator = tf + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                
                # Calculate term score
                term_score = idf * (numerator / denominator if denominator > 0 else 0)
                
                # Add to document score
                score += term_score
        
        return score
    
    #-------------------------------------------------------------------------
    # Utility functions to rank documents for a query
    #-------------------------------------------------------------------------
    def rank_documents(self, query, method='bm25', top_k=10, **kwargs):
        """
        Rank documents for a given query using the specified method.
        
        Args:
            query: Query string
            method: Ranking method ('tfidf', 'cosine', or 'bm25')
            top_k: Number of top documents to return
            **kwargs: Additional parameters for specific ranking methods
            
        Returns:
            List of (doc_idx, score) tuples for top-k documents
        """
        scores = []
        
        for doc_idx in range(self.N):
            if method.lower() == 'tfidf':
                score = self.tfidf_score(query, doc_idx)
            elif method.lower() == 'cosine':
                score = self.cosine_similarity(query, doc_idx)
            elif method.lower() == 'bm25':
                # Extract BM25 parameters if provided
                k1 = kwargs.get('k1', 1.5)
                b = kwargs.get('b', 0.75)
                score = self.bm25_score(query, doc_idx, k1, b)
            else:
                raise ValueError(f"Unknown ranking method: {method}")
            
            scores.append((doc_idx, score))
        
        # Sort by score in descending order and return top-k
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


# Example usage
if __name__ == "__main__":
    # Sample document collection
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A brown fox is quick and jumps over lazy dogs",
        "The dog was lazy and the fox was quick",
        "A document about something completely different like computers",
        "Information retrieval is the process of obtaining information from a collection of resources"
    ]
    
    # Initialize ranking formulas
    ranker = RankingFormulas(documents)
    
    # Sample query
    query = "quick brown fox"
    
    # Compare ranking methods
    print(f"Query: '{query}'\n")
    
    print("TF-IDF Ranking:")
    tfidf_results = ranker.rank_documents(query, method='tfidf')
    for doc_idx, score in tfidf_results:
        print(f"Doc {doc_idx}: {documents[doc_idx][:50]}... (Score: {score:.4f})")
    
    print("\nCosine Similarity Ranking:")
    cosine_results = ranker.rank_documents(query, method='cosine')
    for doc_idx, score in cosine_results:
        print(f"Doc {doc_idx}: {documents[doc_idx][:50]}... (Score: {score:.4f})")
    
    print("\nBM25 Ranking:")
    bm25_results = ranker.rank_documents(query, method='bm25')
    for doc_idx, score in bm25_results:
        print(f"Doc {doc_idx}: {documents[doc_idx][:50]}... (Score: {score:.4f})") 