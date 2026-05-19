# Adversarial QA Probe — Analysis Memo

## 1. Hypothesis

- **Input pattern:**  
Inputs contain multiple named entities of the same type (people, companies, or roles) appearing in close proximity within short factual sentences. Many examples are structured as “A did X while B did Y,” where both A and B are plausible answers.

- **Output pattern:**  
The model often selects the most prominent or earliest-mentioned entity in the context rather than the entity that correctly satisfies the question’s relational constraint (e.g., role-specific or action-specific grounding).

- **Why you hypothesize this:**  
The failure likely comes from the extractive QA model relying on surface-level lexical cues and positional bias rather than relational reasoning. DistilBERT-style QA models are trained to maximize span probability locally, so they over-weight early or salient entities instead of correctly interpreting roles such as founder vs CEO or actor vs director.

---

## 2. Set Design

- **Total examples:** 30  

- **Tags used:**
  - same_type_distractor (10)
  - nearby_entity (13)
  - reversed_roles (5)
  - control_easy (2)

- **Why these tags:**
  - **same_type_distractor:** tests confusion between entities of the same semantic class (e.g., multiple companies or people in similar roles).
  - **nearby_entity:** tests positional bias where the model selects nearby but incorrect entities in the same sentence.
  - **reversed_roles:** tests relational misunderstanding where subject/object roles are swapped (e.g., founder vs CEO).
  - **control_easy:** verifies that the model can succeed when no distractors are present.

- **Control examples:**  
4 control examples were included to confirm that the model performs perfectly when no ambiguity is present. This ensures failures are due to interference patterns, not dataset or evaluation issues.

---

## 3. Results

- **Aggregate EM:** 0.90  
- **Aggregate F1:** 0.9167  

- **Lab 7B baseline:**  
  - EM: 0.3440  
  - F1: 0.4611  

### Per-pattern breakdown:

| Pattern | n | EM | F1 | vs. baseline |
|---|---|---|---|---|
| same_type_distractor | 10 | 0.9 | 0.9 | ↑ higher than baseline |
| nearby_entity | 13 | 0.92 | 0.96 | ↑ higher than baseline |
| reversed_roles | 5 | 0.8 | 0.8 | ↑ higher than baseline |
| control_easy | 2 | 1.00 | 1.00 | ↑ higher than baseline |

### Example failure cases:

- **(S_014)** Who scored the winning goal in the match?  
  - Gold: Lionel Messi  
  - Predicted: Kylian Mbappé  
  - Model confuses multiple star players mentioned in the same match report and selects a nearby high-profile entity.

- **(S_006)** Which team won the championship?  
  - Gold: Manchester City  
  - Predicted: Real Madrid  
  - Model selects the historically dominant team mentioned in context instead of the actual match winner.

- **(S_021)** Who was named MVP of the final game?  
  - Gold: Stephen Curry  
  - Predicted: LeBron James  
  - Role confusion between multiple high-impact players in the same narrative.

---

## 4. Production Defense

A confidence-threshold routing system is the most appropriate mitigation.

Since most errors occur when multiple similar entities compete in the same context, the model should not always be trusted to return an answer. Instead, predictions with low confidence (e.g., small difference between top spans) should be routed to a fallback system such as human review or retrieval-based verification.

This directly addresses the observed failure mode: the model performs well on clean inputs but becomes unreliable under entity competition and role ambiguity. Filtering uncertain predictions improves reliability in production settings where factual correctness is critical.