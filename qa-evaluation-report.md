# QA Evaluation Report

## Dataset Description

The evaluation dataset contains approximately 1,000 extractive question-answering examples derived from the CNN technology and entertainment slice of the glnmario/news-qa-summarization dataset. Each example includes a question, a context passage from a news article, and a gold answer span that appears directly in the context.

## Model

Model: distilbert-base-cased-distilled-squad  

Hugging Face Hub:  
https://huggingface.co/distilbert-base-cased-distilled-squad  

## Aggregate Metrics

Exact Match (EM): 34.40  
Token-F1: 46.11  

The token-F1 score is noticeably higher than the exact-match score, which suggests that the model often produces partially correct spans even when it does not exactly match the gold answer. This indicates that the model is usually able to locate relevant information in the context but struggles with precise boundary selection and exact span extraction.

---

## Failure-Mode Taxonomy

### 1. Distractor Entity Confusion

- qid: NEWS_0195_Q2  
- Question: Jada's daughter will appear in what?  
- Gold Answer: "Karate Kid" remake.  
- Predicted Answer: TNT Network  

In this case, the model selects a related but incorrect entity from the same context. Instead of focusing on the answer to the question, it picks a more prominent or nearby entity, showing confusion between multiple candidates in the passage.

---

### 2. Semantic Drift (Thematic Guessing)

- qid: NEWS_0608_Q2  
- Question: What upgrade was announced?  
- Gold Answer: Apple's iLife and iWork software suites  
- Predicted Answer: iPhoto  

Here, the prediction is related to the general topic but does not correctly answer the question. The model appears to rely on semantic similarity rather than exact span grounding, resulting in partially relevant but incorrect outputs.

---

### 3. Numerical and Span Precision Errors

- qid: NEWS_0408_Q2  
- Question: How many Academy Awards was "Slumdog Millionaire" nominated for?  
- Gold Answer: 10  
- Predicted Answer: three  

This shows a failure in precise extraction, especially for numerical values or exact spans. The model tends to substitute nearby or commonly associated values instead of retrieving the exact correct token span.

---

## Domain Judgment

I would cautiously deploy this model for customer documentation question answering because it provides fast inference and generally produces relevant partial answers. However, I would not deploy it in high-stakes domains such as legal or medical question answering because it lacks reliable boundary precision, does not handle uncertainty or no-answer cases robustly, and may produce confidently incorrect spans when multiple entities are present.