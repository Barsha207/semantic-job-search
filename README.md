# Semantic Job Search

A small demo that finds jobs by meaning, not keywords, using sentence
embeddings (all-MiniLM-L6-v2) and cosine similarity.

## How it works
1. Each job description is converted into an embedding vector.
2. Your query is converted into a vector the same way.
3. Jobs are ranked by cosine similarity to the query.

## Run
pip install -r requirements.txt
python semantic_search.py

## Example
Query: "I like building AI chatbots"
Top match: "LLM engineer building AI agents with tool calling" (score 0.53),
even though the two share no keywords.

## Built with
Python, sentence-transformers, NumPy