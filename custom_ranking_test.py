from ranking_formulas import RankingFormulas
import numpy as np
import matplotlib.pyplot as plt

def print_ranking_results(title, results, documents):
    """Print ranking results in a formatted way."""
    print(f"\n{title}:")
    for i, (doc_idx, score) in enumerate(results, 1):
        print(f"{i}. Doc {doc_idx + 1}: \"{documents[doc_idx][:50]}...\" (Score: {score:.4f})")

def compare_all_methods():
    """Compare all three ranking methods on a common example."""
    # Create a document collection with clear relevance differences
    documents = [
        "Information retrieval systems are used to find relevant documents from a collection.",
        "Document retrieval focuses on finding information relevant to user queries.",
        "Search engines use information retrieval techniques to find relevant web pages.",
        "Machine learning can improve retrieval performance in search systems.",
        "Database systems store and retrieve structured data efficiently.",
        "A completely irrelevant document about cooking and recipes."
    ]
    
    print(f"Document Collection:\n")
    for i, doc in enumerate(documents, 1):
        print(f"Document {i}: \"{doc}\"")
    
    # Create multiple test queries with varying complexity
    queries = [
        "information retrieval",
        "relevant documents",
        "search systems",
        "information retrieval relevant documents",
        "database retrieval"
    ]
    
    # Initialize ranking formulas
    ranker = RankingFormulas(documents)
    
    # Compare all methods for each query
    for query in queries:
        print(f"\n{'='*80}")
        print(f"QUERY: \"{query}\"")
        print(f"{'='*80}")
        
        # Get rankings from each method
        tfidf_results = ranker.rank_documents(query, method="tfidf")
        cosine_results = ranker.rank_documents(query, method="cosine")
        bm25_results = ranker.rank_documents(query, method="bm25")
        
        # Print results
        print_ranking_results("TF-IDF RANKING", tfidf_results, documents)
        print_ranking_results("COSINE SIMILARITY RANKING", cosine_results, documents)
        print_ranking_results("BM25 RANKING", bm25_results, documents)
        
        # Create a visualization comparing the scores
        plt.figure(figsize=(12, 6))
        
        # Get document indices and scores
        doc_indices = np.arange(len(documents))
        
        # Create a mapping of document index to score for each method
        tfidf_scores = np.zeros(len(documents))
        cosine_scores = np.zeros(len(documents))
        bm25_scores = np.zeros(len(documents))
        
        for doc_idx, score in tfidf_results:
            tfidf_scores[doc_idx] = score
        
        for doc_idx, score in cosine_results:
            cosine_scores[doc_idx] = score
        
        for doc_idx, score in bm25_results:
            bm25_scores[doc_idx] = score
        
        # Normalize scores for better comparison
        if np.max(tfidf_scores) > 0:
            tfidf_scores = tfidf_scores / np.max(tfidf_scores)
        if np.max(cosine_scores) > 0:
            cosine_scores = cosine_scores / np.max(cosine_scores)
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
        
        # Plot scores
        width = 0.25
        plt.bar(doc_indices - width, tfidf_scores, width, label='TF-IDF')
        plt.bar(doc_indices, cosine_scores, width, label='Cosine Similarity')
        plt.bar(doc_indices + width, bm25_scores, width, label='BM25')
        
        plt.xlabel('Document')
        plt.ylabel('Normalized Score')
        plt.title(f'Comparison of Ranking Methods for Query: "{query}"')
        plt.xticks(doc_indices, [f'Doc {i+1}' for i in range(len(documents))])
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        clean_query = query.replace(' ', '_')
        plt.savefig(f'ranking_comparison_{clean_query}.png')
        plt.close()
        print(f"\nScore comparison visualization saved as 'ranking_comparison_{clean_query}.png'")
        
        # Print concordances and disagreements
        print("\nCONCORDANCES AND DISAGREEMENTS:")
        
        # Create dictionary mapping doc_idx to rank for each method
        tfidf_ranks = {doc_idx: rank for rank, (doc_idx, _) in enumerate(tfidf_results, 1)}
        cosine_ranks = {doc_idx: rank for rank, (doc_idx, _) in enumerate(cosine_results, 1)}
        bm25_ranks = {doc_idx: rank for rank, (doc_idx, _) in enumerate(bm25_results, 1)}
        
        # Check for complete agreement on top document
        top_docs = [tfidf_results[0][0] if tfidf_results else None, 
                   cosine_results[0][0] if cosine_results else None, 
                   bm25_results[0][0] if bm25_results else None]
        
        if len(set(top_docs)) == 1 and top_docs[0] is not None:
            print(f"- All methods agree on the top document: Document {top_docs[0] + 1}")
        else:
            print("- Methods disagree on the top document:")
            if tfidf_results:
                print(f"  - TF-IDF: Document {tfidf_results[0][0] + 1}")
            if cosine_results:
                print(f"  - Cosine: Document {cosine_results[0][0] + 1}")
            if bm25_results:
                print(f"  - BM25: Document {bm25_results[0][0] + 1}")
        
        # Check for differences in ranking
        print("\nRanking differences:")
        for doc_idx in range(len(documents)):
            tfidf_rank = tfidf_ranks.get(doc_idx, "Not ranked")
            cosine_rank = cosine_ranks.get(doc_idx, "Not ranked")
            bm25_rank = bm25_ranks.get(doc_idx, "Not ranked")
            
            if tfidf_rank != cosine_rank or tfidf_rank != bm25_rank or cosine_rank != bm25_rank:
                print(f"- Document {doc_idx + 1} is ranked:")
                print(f"  - {tfidf_rank} by TF-IDF")
                print(f"  - {cosine_rank} by Cosine Similarity")
                print(f"  - {bm25_rank} by BM25")
        
        # Continue to next query?
        if query != queries[-1]:
            print("\nPress Enter to continue to the next query...")
            input()

if __name__ == "__main__":
    try:
        import matplotlib
    except ImportError:
        print("Installing matplotlib...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
    
    compare_all_methods() 