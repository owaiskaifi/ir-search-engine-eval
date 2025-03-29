import os
import json
import time
from ranking_formulas import RankingFormulas
from tabulate import tabulate

def load_documents():
    """Load CISI documents for testing the ranking formulas."""
    print("Loading CISI documents...")
    
    with open('output/documents.json', 'r') as f:
        documents_meta = json.load(f)
    
    # Create a list of document texts
    documents = []
    doc_ids = []
    
    for doc_id, doc_info in documents_meta.items():
        # Combine title, author and content
        doc_text = f"{doc_info['title']} {doc_info['author']} {doc_info['content']}"
        documents.append(doc_text)
        doc_ids.append(doc_id)
    
    print(f"Loaded {len(documents)} documents.")
    return documents, doc_ids

def test_ranking_formulas():
    """Test the ranking formulas on the CISI dataset."""
    # Check if the documents have been indexed
    if not os.path.exists('output/documents.json'):
        print("Please run 'python main.py' first to index the CISI dataset.")
        return
    
    # Load documents
    documents, doc_ids = load_documents()
    
    # Initialize ranking formulas
    print("Initializing ranking formulas (this may take a few minutes)...")
    start_time = time.time()
    ranker = RankingFormulas(documents)
    print(f"Initialization completed in {time.time() - start_time:.2f} seconds.")
    
    # Test queries
    queries = [
        "information retrieval",
        "document ranking",
        "search engine",
        "database management",
        "library science"
    ]
    
    # Run ranking methods for each query
    for query in queries:
        print(f"\n=== Query: '{query}' ===")
        
        # Run and time each ranking method
        methods = ['tfidf', 'cosine', 'bm25']
        results = {}
        
        for method in methods:
            start_time = time.time()
            method_results = ranker.rank_documents(query, method=method, top_k=5)
            elapsed_time = time.time() - start_time
            
            # Map document indices to actual document IDs
            mapped_results = [(doc_ids[idx], score) for idx, score in method_results]
            results[method] = (mapped_results, elapsed_time)
        
        # Display results in a table
        table_data = []
        headers = ["Rank", "TF-IDF Doc ID", "TF-IDF Score", "Cosine Doc ID", "Cosine Score", "BM25 Doc ID", "BM25 Score"]
        
        for i in range(min(5, max(len(results[m][0]) for m in methods))):
            row = [i + 1]
            
            for method in methods:
                method_results, _ = results[method]
                
                if i < len(method_results):
                    doc_id, score = method_results[i]
                    row.extend([doc_id, f"{score:.4f}"])
                else:
                    row.extend(["-", "-"])
            
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Display timing information
        timing_data = []
        for method in methods:
            _, elapsed_time = results[method]
            timing_data.append([method.upper(), f"{elapsed_time:.4f} seconds"])
        
        print("\nRuntime Comparison:")
        print(tabulate(timing_data, headers=["Method", "Time"], tablefmt="grid"))
        
        # Print details of top document from each method
        print("\nTop Document Details:")
        for method in methods:
            method_results, _ = results[method]
            if method_results:
                top_doc_id, top_score = method_results[0]
                top_doc_idx = doc_ids.index(top_doc_id)
                top_doc_text = documents[top_doc_idx]
                print(f"\n{method.upper()} Top Result (ID: {top_doc_id}, Score: {top_score:.4f}):")
                print(f"Text: {top_doc_text[:200]}..." if len(top_doc_text) > 200 else top_doc_text)
        
        # Ask if user wants to continue
        if query != queries[-1]:
            choice = input("\nPress Enter to continue to the next query, or 'q' to quit: ")
            if choice.lower() == 'q':
                break

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("The tabulate package is required. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "tabulate"])
        from tabulate import tabulate
    
    test_ranking_formulas() 