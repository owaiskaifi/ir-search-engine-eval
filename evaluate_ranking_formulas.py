import os
import json
import numpy as np
import matplotlib.pyplot as plt
from ranking_formulas import RankingFormulas
from tabulate import tabulate
import math
import time
from collections import defaultdict

class RankingEvaluator:
    """
    Evaluates ranking formulas against relevance judgments using standard IR metrics:
    - Precision
    - Recall
    - nDCG (Normalized Discounted Cumulative Gain)
    - MRR (Mean Reciprocal Rank)
    - Average Precision
    - MAP (Mean Average Precision)
    """
    
    def __init__(self, documents_path='output/documents.json', 
                 queries_path='output/queries.json', 
                 relevance_path='data/CISI.REL'):
        """
        Initialize the evaluator with documents, queries, and relevance judgments.
        
        Args:
            documents_path: Path to the JSON file containing documents
            queries_path: Path to the JSON file containing queries
            relevance_path: Path to the relevance judgments file
        """
        self.documents_path = documents_path
        self.queries_path = queries_path
        self.relevance_path = relevance_path
        
        # Load data
        self.load_documents()
        self.load_queries()
        self.load_relevance_judgments()
        
        # Initialize ranking formulas
        print("Initializing ranking formulas (this may take a few minutes)...")
        start_time = time.time()
        self.ranker = RankingFormulas([doc['text'] for doc in self.documents.values()])
        print(f"Initialization completed in {time.time() - start_time:.2f} seconds.")
    
    def load_documents(self):
        """Load CISI documents."""
        print("Loading documents...")
        
        try:
            with open(self.documents_path, 'r') as f:
                self.documents = json.load(f)
            
            # Convert document IDs to strings if they're not already
            if not all(isinstance(key, str) for key in self.documents.keys()):
                self.documents = {str(k): v for k, v in self.documents.items()}
            
            # Add text field that combines title, author and content
            for doc_id, doc_info in self.documents.items():
                doc_info['text'] = f"{doc_info['title']} {doc_info['author']} {doc_info['content']}"
            
            # Create document ID mapping (for converting between indices and IDs)
            self.doc_ids = list(self.documents.keys())
            self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
            
            print(f"Loaded {len(self.documents)} documents.")
        except Exception as e:
            print(f"Error loading documents: {e}")
            self.documents = {}
            self.doc_ids = []
            self.doc_id_to_idx = {}
    
    def load_queries(self):
        """Load CISI queries."""
        print("Loading queries...")
        
        try:
            with open(self.queries_path, 'r') as f:
                self.queries = json.load(f)
            
            # Convert query IDs to strings if they're not already
            if not all(isinstance(key, str) for key in self.queries.keys()):
                self.queries = {str(k): v for k, v in self.queries.items()}
            
            print(f"Loaded {len(self.queries)} queries.")
        except Exception as e:
            print(f"Error loading queries: {e}")
            self.queries = {}
    
    def load_relevance_judgments(self):
        """Load CISI relevance judgments."""
        print("Loading relevance judgments...")
        
        try:
            # Parse relevance judgments file (format: query_id doc_id 0 0.0)
            self.relevance = defaultdict(set)
            
            with open(self.relevance_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        query_id = parts[0].strip()
                        doc_id = parts[1].strip()
                        self.relevance[query_id].add(doc_id)
            
            # Count total number of judgments
            total_judgments = sum(len(docs) for docs in self.relevance.values())
            print(f"Loaded {total_judgments} relevance judgments for {len(self.relevance)} queries.")
        except Exception as e:
            print(f"Error loading relevance judgments: {e}")
            self.relevance = defaultdict(set)
    
    def evaluate_ranking(self, method='bm25', k=10, num_queries=None):
        """
        Evaluate a ranking method on the test collection.
        
        Args:
            method: Ranking method to use ('tfidf', 'cosine', or 'bm25')
            k: Number of top documents to retrieve
            num_queries: Number of queries to evaluate (None = all queries)
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.documents or not self.queries or not self.relevance:
            print("Error: Documents, queries, or relevance judgments not loaded.")
            return None
        
        # Metrics to track
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        mrr_values = []
        ap_values = []
        
        # Filter to queries that have relevance judgments
        eval_query_ids = [qid for qid in self.relevance if qid in self.queries]
        
        # Limit number of queries if specified
        if num_queries and num_queries < len(eval_query_ids):
            eval_query_ids = eval_query_ids[:num_queries]
        
        # Evaluate each query
        for query_id in eval_query_ids:
            query = self.queries[query_id]['text']
            relevant_docs = self.relevance[query_id]
            
            # Skip if no relevant documents
            if not relevant_docs:
                continue
            
            # Get all document indices for conversion
            doc_indices = range(len(self.doc_ids))
            
            # Rank documents
            start_time = time.time()
            results = self.ranker.rank_documents(query, method=method, top_k=k)
            query_time = time.time() - start_time
            
            # Extract document IDs from results
            retrieved_doc_ids = [self.doc_ids[idx] for idx, _ in results]
            
            # Calculate metrics
            precision = self.precision(retrieved_doc_ids, relevant_docs)
            recall = self.recall(retrieved_doc_ids, relevant_docs)
            ndcg = self.ndcg(retrieved_doc_ids, relevant_docs, k)
            rr = self.reciprocal_rank(retrieved_doc_ids, relevant_docs)
            ap = self.average_precision(retrieved_doc_ids, relevant_docs)
            
            # Store metrics
            precision_at_k.append(precision)
            recall_at_k.append(recall)
            ndcg_at_k.append(ndcg)
            mrr_values.append(rr)
            ap_values.append(ap)
        
        # Calculate mean metrics
        mean_precision = np.mean(precision_at_k) if precision_at_k else 0
        mean_recall = np.mean(recall_at_k) if recall_at_k else 0
        mean_ndcg = np.mean(ndcg_at_k) if ndcg_at_k else 0
        mrr = np.mean(mrr_values) if mrr_values else 0
        map_value = np.mean(ap_values) if ap_values else 0
        
        # Return evaluation results
        return {
            'method': method,
            'num_queries': len(eval_query_ids),
            'precision': mean_precision,
            'recall': mean_recall,
            'ndcg': mean_ndcg,
            'mrr': mrr,
            'map': map_value,
            'precision_per_query': precision_at_k,
            'recall_per_query': recall_at_k,
            'ndcg_per_query': ndcg_at_k,
            'mrr_per_query': mrr_values,
            'ap_per_query': ap_values,
        }
    
    def precision(self, retrieved_docs, relevant_docs):
        """
        Calculate precision at k.
        
        Precision = |{relevant_docs} ∩ {retrieved_docs}| / |{retrieved_docs}|
        """
        if not retrieved_docs:
            return 0
        
        relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
        return relevant_retrieved / len(retrieved_docs)
    
    def recall(self, retrieved_docs, relevant_docs):
        """
        Calculate recall at k.
        
        Recall = |{relevant_docs} ∩ {retrieved_docs}| / |{relevant_docs}|
        """
        if not relevant_docs:
            return 0
        
        relevant_retrieved = len(set(retrieved_docs) & set(relevant_docs))
        return relevant_retrieved / len(relevant_docs)
    
    def ndcg(self, retrieved_docs, relevant_docs, k):
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        nDCG@k = DCG@k / IDCG@k
        
        where DCG@k = ∑(i=1 to k) rel_i / log_2(i+1)
        and IDCG@k is the DCG@k of the ideal ranking
        """
        if not relevant_docs or not retrieved_docs:
            return 0
        
        # For binary relevance, rel_i is 1 if the document is relevant, 0 otherwise
        dcg = 0
        for i, doc_id in enumerate(retrieved_docs):
            rel = 1 if doc_id in relevant_docs else 0
            # Add 1 to i because i is 0-indexed but the formula is 1-indexed
            dcg += rel / math.log2(i + 2)
        
        # Calculate ideal DCG (when all top documents are relevant)
        idcg = 0
        for i in range(min(len(relevant_docs), k)):
            idcg += 1 / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0
    
    def reciprocal_rank(self, retrieved_docs, relevant_docs):
        """
        Calculate the reciprocal rank (1/rank of the first relevant document).
        
        RR = 1 / rank of first relevant doc
        """
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                # Add 1 to i because i is 0-indexed but ranks are 1-indexed
                return 1 / (i + 1)
        return 0
    
    def average_precision(self, retrieved_docs, relevant_docs):
        """
        Calculate Average Precision.
        
        AP = ∑(k=1 to n) (P@k * rel_k) / |{relevant_docs}|
        
        where rel_k is 1 if the kth document is relevant, 0 otherwise
        """
        if not relevant_docs:
            return 0
        
        hits = 0
        sum_precisions = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                hits += 1
                # Precision at the current position
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        # Average precision = sum of precisions / number of relevant documents
        return sum_precisions / len(relevant_docs)
    
    def compare_methods(self, k=10, num_queries=None):
        """
        Compare all ranking methods and display results.
        
        Args:
            k: Number of top documents to retrieve
            num_queries: Number of queries to evaluate (None = all queries)
        """
        methods = ['tfidf', 'cosine', 'bm25']
        results = []
        
        print(f"\nEvaluating ranking methods with k={k}...")
        
        for method in methods:
            print(f"Evaluating {method.upper()}...")
            eval_result = self.evaluate_ranking(method, k, num_queries)
            results.append(eval_result)
        
        # Create a summary table
        headers = ["Method", "Precision", "Recall", "nDCG", "MRR", "MAP"]
        table_data = []
        
        for result in results:
            table_data.append([
                result['method'].upper(),
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['ndcg']:.4f}",
                f"{result['mrr']:.4f}",
                f"{result['map']:.4f}"
            ])
        
        print("\nPerformance Comparison:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Create visualizations
        self.visualize_comparison(results)
        
        return results
    
    def visualize_comparison(self, results):
        """Create visualizations comparing the different ranking methods."""
        methods = [r['method'].upper() for r in results]
        metrics = ['precision', 'recall', 'ndcg', 'mrr', 'map']
        metric_labels = ['Precision', 'Recall', 'nDCG', 'MRR', 'MAP']
        
        # Bar chart for main metrics
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, result in enumerate(results):
            values = [result[metric] for metric in metrics]
            plt.bar(x + i*width, values, width, label=result['method'].upper())
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Comparison of Ranking Methods')
        plt.xticks(x + width, metric_labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig('ranking_metrics_comparison.png')
        print("Visualization saved as 'ranking_metrics_comparison.png'")
        
        # Box plots for per-query performance
        plt.figure(figsize=(14, 10))
        
        # Map metrics to their respective per-query keys
        metric_to_key = {
            'precision': 'precision_per_query',
            'recall': 'recall_per_query',
            'ndcg': 'ndcg_per_query',
            'mrr': 'mrr_per_query',
            'map': 'ap_per_query'  # Fix: 'map' uses 'ap_per_query' for per-query values
        }
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            
            data = [result[metric_to_key[metric]] for result in results]
            plt.boxplot(data, labels=methods)
            
            plt.title(f'{metric_labels[i]} Distribution')
            plt.ylabel('Score')
            plt.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('ranking_metrics_distribution.png')
        print("Distribution visualization saved as 'ranking_metrics_distribution.png'")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("The tabulate package is required. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "tabulate"])
        from tabulate import tabulate
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
    
    # Create evaluator
    evaluator = RankingEvaluator()
    
    # Evaluate all methods for only k=10 and limit to 20 queries for faster testing
    evaluator.compare_methods(k=10, num_queries=20) 