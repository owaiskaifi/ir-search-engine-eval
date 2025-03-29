# CISI Information Retrieval System

A comprehensive information retrieval system built for the CISI document collection, featuring multiple ranking algorithms and evaluation tools.

## Overview

This project implements and evaluates different ranking formulas for information retrieval on the CISI (Computer and Information Science Abstracts) dataset. The system provides:

1. Tools for indexing and preprocessing the CISI document collection
2. Implementation of three major ranking formulas:
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Cosine Similarity
   - BM25 (Okapi BM25)
3. Comprehensive evaluation of ranking performance using standard IR metrics
4. Visualization and analysis tools for comparing different ranking methods

## Dataset

The CISI dataset is a standard test collection for information retrieval research and contains:
- 1,460 documents from the computer science and information science domains
- 112 queries
- 3,114 relevance judgments that map queries to relevant documents

## Project Structure

- `main.py`: Main script for processing and indexing the CISI dataset
- `search.py`: Basic search functionality using TF-IDF and cosine similarity
- `ranking.py`: Enhanced search with multiple ranking methods
- `ranking_formulas.py`: Standalone implementation of the three ranking formulas
- `evaluate_ranking_formulas.py`: Evaluation script for measuring performance
- `compare_rankings.py`: Tool for comparing different ranking methods side by side
- `custom_ranking_test.py`: Custom tests with simplified examples
- `test_ranking_formulas.py`: Tests using the CISI dataset
- Data files in `/data`:
  - `CISI.ALL`: Document collection
  - `CISI.QRY`: Queries
  - `CISI.REL`: Relevance judgments

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Indexing the Dataset

```bash
python main.py
```

This processes the CISI dataset and creates index files in the `output` directory.

### Basic Search

```bash
python search.py
```

Provides a simple search interface using TF-IDF with cosine similarity.

### Advanced Search with Multiple Ranking Methods

```bash
python ranking.py
```

Performs search using all three ranking methods and allows comparison.

### Evaluating Ranking Performance

```bash
python evaluate_ranking_formulas.py
```

Evaluates the performance of all ranking methods using standard IR metrics:
- Precision
- Recall
- nDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)

### Comparing Rankings

```bash
python compare_rankings.py
```

Provides a side-by-side comparison of how different methods rank the same documents.

### Testing with Simplified Examples

```bash
python custom_ranking_test.py
```

Demonstrates the ranking methods on a small set of documents for better understanding.

## Ranking Formulas

### TF-IDF (Term Frequency-Inverse Document Frequency)

A statistical measure that evaluates how important a word is to a document relative to a collection:
- **Term Frequency (TF)**: How frequently a term appears in a document
- **Inverse Document Frequency (IDF)**: How rare or common a term is across all documents

```
score(q, d) = ∑(tf(t, d) * idf(t))
```

### Cosine Similarity

Measures the cosine of the angle between two non-zero vectors, determining similarity between documents and queries in a vector space model:

```
similarity(q, d) = (q · d) / (||q|| * ||d||)
```

### BM25 (Okapi BM25)

A probabilistic ranking function that extends TF-IDF with term saturation and document length normalization:

```
score(q, d) = ∑ IDF(t) * (f(t, d) * (k1 + 1)) / (f(t, d) + k1 * (1 - b + b * |d|/avgdl))
```

Where:
- k1: Controls term frequency saturation (typically between 1.2 and 2.0)
- b: Controls document length normalization (typically 0.75)

## Evaluation Results

Our evaluation on the CISI dataset shows the following performance for the top 10 documents:

| Method | Precision | Recall | nDCG | MRR | MAP |
|--------|-----------|--------|------|-----|-----|
| TF-IDF | 0.1950 | 0.0592 | 0.2111 | 0.3667 | 0.0285 |
| COSINE | 0.2650 | 0.0784 | 0.2599 | 0.3892 | 0.0409 |
| BM25 | 0.3050 | 0.1510 | 0.3472 | 0.5919 | 0.0667 |

BM25 consistently outperforms the other methods across all metrics, with Cosine Similarity offering a good balance between performance and simplicity. Detailed evaluation results and visualizations are available in the `ranking_evaluation_report.md` file.

## Visualizations

The project generates several visualizations to aid in understanding the performance differences between ranking methods:

- `ranking_metrics_comparison.png`: Bar chart comparing all metrics across methods
- `ranking_metrics_distribution.png`: Box plots showing the distribution of performance across queries
- Various comparison charts for specific queries in the `custom_ranking_test.py` script

## Requirements

- Python 3.7+
- numpy
- pandas
- nltk
- scikit-learn
- scipy
- matplotlib
- tabulate

## Future Work

- Parameter optimization for BM25
- Implementation of query expansion techniques
- Development of ensemble ranking methods
- Integration of relevance feedback mechanisms
- Testing with additional document collections

## References

1. Manning, C.D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.
2. Robertson, S.E., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 333-389.
3. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513-523. 