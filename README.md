# tasteprofile
 
An end-to-end food-recommendation RAG system: an offline pipeline enriches and embeds a recipe dataset into a Chroma vector store, and a Streamlit app serves two modes on top of it — an LLM-driven conversational search that parses free-text queries into filters and answers grounded strictly in retrieved foods, and a non-LLM like/dislike recommender that re-ranks candidates by vector similarity and ingredient/taste/cuisine overlap.
 
## Screenshots
 
**Chat Search** — parses a free-text query into filters, retrieves matches from Chroma, and answers using only those retrieved foods.
![Chat Search](docs/screenshots/chat-search.png)
 
**Like / Dislike Recommender** — re-ranks foods against your picks and returns a match score with reasoning, no LLM involved.

![Like / Dislike Recommender](docs/screenshots/recommender.png)
 
## Eval results
 
| Metric | Score |
|---|---|
| Rule-based pass rate (50 cases) | 41/50 (82%) |
| RAGAS Faithfulness | 0.825 |
| RAGAS Answer Relevancy | 0.793 |
| RAGAS Context Precision | 0.988 |
| DeepEval Answer Relevancy | 0.886 |
| DeepEval Faithfulness | 0.758 |
| DeepEval Contextual Precision | 0.912 |
| DeepEval Contextual Recall | 0.762 |
 
The system is strong at retrieving the *right* context (context precision >0.9 on both frameworks) but weaker at keeping generated answers fully grounded in that context (faithfulness sits in the mid-0.7s to low-0.8s), making faithfulness the top target for improvement.
