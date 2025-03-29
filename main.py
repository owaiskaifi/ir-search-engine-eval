import os
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pickle
import json

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class CISIDataset:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.docs_file = os.path.join(data_dir, 'CISI.ALL')
        self.query_file = os.path.join(data_dir, 'CISI.QRY')
        self.rel_file = os.path.join(data_dir, 'CISI.REL')
        
        # Initialize containers
        self.documents = {}
        self.queries = {}
        self.relevance = {}
        
        # For preprocessing
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # For indexing
        self.vectorizer = None
        self.doc_vectors = None
        self.doc_ids = []
        
    def parse_cisi_all(self):
        """Parse the CISI.ALL file to extract documents."""
        print("Parsing documents...")
        
        with open(self.docs_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split the file into documents based on .I tag
        docs = re.split(r'\.I\s+', content)[1:]  # Skip first empty element
        
        for doc in docs:
            # Extract document ID
            doc_id = doc.split('\n')[0].strip()
            
            # Extract title
            title_match = re.search(r'\.T\s+(.*?)(?=\.[A-Z])', doc, re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""
            
            # Extract author
            author_match = re.search(r'\.A\s+(.*?)(?=\.[A-Z])', doc, re.DOTALL)
            author = author_match.group(1).strip() if author_match else ""
            
            # Extract content
            content_match = re.search(r'\.W\s+(.*?)(?=\.[A-Z]|$)', doc, re.DOTALL)
            content = content_match.group(1).strip() if content_match else ""
            
            # Store the document
            self.documents[doc_id] = {
                'id': doc_id,
                'title': title,
                'author': author, 
                'content': content,
                'full_text': f"{title} {author} {content}"
            }
            
        print(f"Parsed {len(self.documents)} documents.")
    
    def parse_cisi_qry(self):
        """Parse the CISI.QRY file to extract queries."""
        print("Parsing queries...")
        
        with open(self.query_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split the file into queries based on .I tag
        queries = re.split(r'\.I\s+', content)[1:]  # Skip first empty element
        
        for query in queries:
            # Extract query ID
            query_id = query.split('\n')[0].strip()
            
            # Extract query text
            query_match = re.search(r'\.W\s+(.*?)(?=\.[A-Z]|$)', query, re.DOTALL)
            query_text = query_match.group(1).strip() if query_match else ""
            
            # Store the query
            self.queries[query_id] = {
                'id': query_id,
                'text': query_text
            }
            
        print(f"Parsed {len(self.queries)} queries.")
    
    def parse_cisi_rel(self):
        """Parse the CISI.REL file to extract relevance judgments."""
        print("Parsing relevance judgments...")
        
        with open(self.rel_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    query_id, doc_id = parts[0], parts[1]
                    
                    if query_id not in self.relevance:
                        self.relevance[query_id] = []
                    
                    self.relevance[query_id].append(doc_id)
        
        print(f"Parsed relevance judgments for {len(self.relevance)} queries.")
    
    def preprocess_text(self, text):
        """Preprocess the text by tokenizing, removing stopwords and stemming."""
        # Handle None or empty text
        if not text:
            return ""
            
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphanumeric tokens
        tokens = [t for t in tokens if t not in self.stop_words and t.isalnum()]
        
        # Apply stemming
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_documents(self):
        """Preprocess all documents in the collection."""
        print("Preprocessing documents...")
        
        for doc_id, doc in self.documents.items():
            doc['preprocessed_text'] = self.preprocess_text(doc['full_text'])
        
        print("Document preprocessing completed.")
    
    def preprocess_queries(self):
        """Preprocess all queries."""
        print("Preprocessing queries...")
        
        for query_id, query in self.queries.items():
            query['preprocessed_text'] = self.preprocess_text(query['text'])
        
        print("Query preprocessing completed.")
    
    def build_index(self):
        """Build a TF-IDF index for the documents."""
        print("Building TF-IDF index...")
        
        # Prepare document list
        preprocessed_docs = []
        self.doc_ids = []
        
        for doc_id, doc in self.documents.items():
            preprocessed_docs.append(doc['preprocessed_text'])
            self.doc_ids.append(doc_id)
        
        # Initialize and fit the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform(preprocessed_docs)
        
        print(f"Indexed {len(preprocessed_docs)} documents with {len(self.vectorizer.get_feature_names_out())} unique terms.")
    
    def save_index(self, output_dir='output'):
        """Save the index and other processed data."""
        print("Saving indexed data...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the vectorizer
        with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save the document vectors
        with open(os.path.join(output_dir, 'doc_vectors.npz'), 'wb') as f:
            np.savez(f, data=self.doc_vectors.data, indices=self.doc_vectors.indices, 
                    indptr=self.doc_vectors.indptr, shape=self.doc_vectors.shape)
        
        # Save document IDs mapping
        with open(os.path.join(output_dir, 'doc_ids.json'), 'w') as f:
            json.dump(self.doc_ids, f)
        
        # Save processed documents (metadata only to save space)
        docs_meta = {doc_id: {k: v for k, v in doc.items() if k != 'preprocessed_text'} 
                     for doc_id, doc in self.documents.items()}
        with open(os.path.join(output_dir, 'documents.json'), 'w') as f:
            json.dump(docs_meta, f)
        
        # Save processed queries
        with open(os.path.join(output_dir, 'queries.json'), 'w') as f:
            json.dump(self.queries, f)
        
        # Save relevance judgments
        with open(os.path.join(output_dir, 'relevance.json'), 'w') as f:
            json.dump(self.relevance, f)
        
        print(f"Saved all indexed data to {output_dir}/")
    
    def process_dataset(self):
        """Process the entire CISI dataset."""
        # Parse the dataset files
        self.parse_cisi_all()
        self.parse_cisi_qry()
        self.parse_cisi_rel()
        
        # Preprocess documents and queries
        self.preprocess_documents()
        self.preprocess_queries()
        
        # Build the index
        self.build_index()
        
        # Save the processed data
        self.save_index()
        
        return self.documents, self.queries, self.relevance

def main():
    # Initialize the dataset processor
    cisi = CISIDataset()
    
    # Process the dataset
    docs, queries, relevance = cisi.process_dataset()
    
    print(f"\nProcessed {len(docs)} documents, {len(queries)} queries, and relevance judgments for {len(relevance)} queries.")
    print("The indexed data is saved in the 'output' directory.")

if __name__ == "__main__":
    main()
