import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ranking import RankingMethods
import pandas as pd
from tabulate import tabulate

class RankingEvaluator:
    def __init__(self):
        self.ranker = RankingMethods()
        
    def run_comparison(self, query, top_k=10):
        """Run a comparison of different ranking methods for the given query."""
        print(f"\nProcessing query: '{query}'")
        
        # Get results from each ranking method
        results_tfidf = self.ranker.rank_tfidf_cosine(query, top_k)
        results_bm25 = self.ranker.rank_bm25(query, top_k)
        results_boolean = self.ranker.rank_boolean_extended(query, top_k)
        
        # Try the proximity ranking if we have multiple terms
        if len(query.split()) > 1:
            results_proximity = self.ranker.rank_proximity(query, top_k)
            methods = [
                ("TF-IDF Cosine", results_tfidf),
                ("BM25", results_bm25),
                ("Extended Boolean", results_boolean),
                ("Term Proximity", results_proximity)
            ]
        else:
            methods = [
                ("TF-IDF Cosine", results_tfidf),
                ("BM25", results_bm25),
                ("Extended Boolean", results_boolean)
            ]
        
        return methods
    
    def display_comparison_table(self, methods, max_results=5):
        """Display a side-by-side comparison table of the ranking results."""
        # Create a table of document IDs and scores for each method
        method_names = [name for name, _ in methods]
        
        # Get all unique document IDs from all methods
        all_doc_ids = set()
        for _, results in methods:
            for result in results[:max_results]:
                all_doc_ids.add(result['id'])
                
        # Create a table with scores and ranks
        table_data = []
        
        # Sort document IDs by their best rank across all methods
        doc_id_best_rank = {}
        for doc_id in all_doc_ids:
            best_rank = float('inf')
            for _, results in methods:
                for i, result in enumerate(results[:max_results], 1):
                    if result['id'] == doc_id and i < best_rank:
                        best_rank = i
            doc_id_best_rank[doc_id] = best_rank if best_rank != float('inf') else max_results + 1
            
        sorted_doc_ids = sorted(all_doc_ids, key=lambda x: doc_id_best_rank[x])
        
        # Create table headers
        headers = ["Doc ID", "Title"]
        for method_name in method_names:
            headers.extend([f"{method_name} Rank", f"{method_name} Score"])
        
        # Fill in the table data
        for doc_id in sorted_doc_ids:
            doc_info = None
            row = [doc_id]
            
            # For each method, add rank and score
            for method_idx, (_, results) in enumerate(methods):
                # Find the document in the results
                rank = "-"
                score = "-"
                for i, result in enumerate(results[:max_results], 1):
                    if result['id'] == doc_id:
                        rank = i
                        score = f"{result['score']:.4f}"
                        doc_info = result
                        break
                
                # Add rank and score to the row
                if method_idx == 0:  # Add title after doc ID only once
                    title = doc_info['title'] if doc_info else "Unknown"
                    row.append(title)
                
                row.extend([rank, score])
            
            table_data.append(row)
        
        # Print the table
        print("\nComparison of Ranking Methods:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def plot_score_comparison(self, methods, max_results=5):
        """Plot a comparison of scores and ranks for each method."""
        method_names = [name for name, _ in methods]
        num_methods = len(methods)
        
        # Get all document IDs and titles that appear in any method's results
        all_docs = {}
        for method_idx, (_, results) in enumerate(methods):
            for i, result in enumerate(results[:max_results], 1):
                doc_id = result['id']
                if doc_id not in all_docs:
                    all_docs[doc_id] = {
                        'title': result['title'],
                        'scores': [0] * num_methods,
                        'ranks': [0] * num_methods
                    }
                all_docs[doc_id]['scores'][method_idx] = result['score']
                all_docs[doc_id]['ranks'][method_idx] = i
        
        # Sort documents by their best rank
        def get_best_rank(doc_data):
            ranks = [r for r in doc_data['ranks'] if r > 0]
            return min(ranks) if ranks else float('inf')
        
        sorted_docs = sorted(all_docs.items(), key=lambda x: get_best_rank(x[1]))
        sorted_docs = sorted_docs[:max_results]  # Limit to top documents
        
        # Prepare data for plotting
        doc_ids = [f"{doc_id}: {data['title'][:20]}..." for doc_id, data in sorted_docs]
        method_scores = []
        method_ranks = []
        
        for method_idx in range(num_methods):
            scores = [data['scores'][method_idx] for _, data in sorted_docs]
            ranks = [data['ranks'][method_idx] for _, data in sorted_docs]
            method_scores.append(scores)
            method_ranks.append(ranks)
        
        # Create a figure with two subplots
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Scores subplot
        ax1 = fig.add_subplot(gs[0])
        bar_width = 0.2
        index = np.arange(len(doc_ids))
        
        for i, scores in enumerate(method_scores):
            ax1.bar(index + i * bar_width, scores, bar_width, label=method_names[i])
        
        ax1.set_title('Score Comparison Across Ranking Methods')
        ax1.set_ylabel('Score')
        ax1.set_xticks(index + bar_width * (num_methods / 2 - 0.5))
        ax1.set_xticklabels(doc_ids, rotation=45, ha='right')
        ax1.legend()
        
        # Ranks subplot
        ax2 = fig.add_subplot(gs[1])
        
        for i, ranks in enumerate(method_ranks):
            # Replace 0 with max_results+1 for documents not in this method's results
            ranks = [r if r > 0 else max_results+1 for r in ranks]
            ax2.plot(index, ranks, 'o-', label=method_names[i])
        
        ax2.set_title('Rank Comparison Across Ranking Methods')
        ax2.set_ylabel('Rank')
        ax2.set_xlabel('Documents')
        ax2.set_xticks(index)
        ax2.set_xticklabels(doc_ids, rotation=45, ha='right')
        ax2.set_ylim(0, max_results + 2)
        ax2.invert_yaxis()  # Lower rank numbers are better
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('ranking_comparison.png')
        print("\nComparison plot saved as 'ranking_comparison.png'")
        plt.close()

def main():
    # Check if tabulate is installed
    try:
        import tabulate
    except ImportError:
        print("The tabulate package is required. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "tabulate"])
        import tabulate
    
    # Check if matplotlib is installed
    try:
        import matplotlib
    except ImportError:
        print("The matplotlib package is required. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib"])
        import matplotlib
    
    evaluator = RankingEvaluator()
    
    print("CISI Ranking Method Comparison Tool")
    print("-----------------------------------")
    print("This tool compares different ranking methods on the CISI dataset.")
    
    while True:
        query = input("\nEnter a search query (or 'quit' to exit): ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        # Run comparison
        methods = evaluator.run_comparison(query)
        
        # Display results
        evaluator.display_comparison_table(methods)
        
        # Plot results
        try:
            evaluator.plot_score_comparison(methods)
        except Exception as e:
            print(f"Could not generate plot: {e}")
        
        # Allow user to see detailed results for a specific method
        while True:
            print("\nAvailable methods:")
            for i, (method_name, _) in enumerate(methods, 1):
                print(f"{i}. {method_name}")
            
            choice = input("\nSelect a method to see detailed results (or press Enter to continue): ")
            
            if not choice:
                break
            
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(methods):
                    method_name, results = methods[choice_idx]
                    print(f"\nDetailed results for {method_name}:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. [{result['id']}] {result['title']} (Score: {result['score']:.4f})")
                        print(f"   Author: {result['author']}")
                        print(f"   {result['snippet']}")
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main() 