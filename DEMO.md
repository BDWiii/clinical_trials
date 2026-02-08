# Why the Retrieval is OP (Overpowered) 🚀

Our retrieval system outperforms standard keyword search by combining **Semantic Embedding** with **Metadata Filtering** and **LLM-Based Reranking**. Here's why it works so effectively on our data.

## 1. The Data Structure
We split our clinical trial data into two distinct parts for the vector store:

```json
{
    "text_data": {
      "Brief Summary": "Frailty and muscle health...",
      "Study Status": "NOT_YET_RECRUITING",
      "Conditions": "Cirrhosis|Frailty|Sarcopenia"
    },
    "meta_data": {
      "NCT Number": "NCT07029243",
      "Study Status": "NOT_YET_RECRUITING",
      "Age": "ADULT, OLDER_ADULT",
      ...
    }
}
```

*   **`text_data` (Embedded):** Contains the *meaningful* narrative (Summary, Conditions). This is what we convert into vectors.
*   **`meta_data` (Not Embedded):** Kept raw for precise filtering (e.g., "Must be RECRUITING", "Must match ADULT").

## 2. Why this is "OP"

### 🧠 Semantic Understanding (vs. Keywords)
*   **Conceptual Matching:** The embeddings (using `all-mpnet-base-v2`) understand that "liver failure" is related to "Cirrhosis" even if the exact words differ.
*   **Context Awareness:** It matches the *intent* of a study (e.g., "maintenance of muscle health") with patient needs, not just random word hits.

### 🎯 Hybrid Approach
*   **Step 1: Broad Semantic Search (Retriever)**
    *   Finds studies that *mean* the right thing.
    *   *Reference:* `tools/retrieval_tool.py` uses ChromaDB with similarity search.
*   **Step 2: Strict Validation (Validator)**
    *   Checks the `meta_data` to ensure the patient actually fits (Age, Sex, Status).
*   **Step 3: Intelligent Scoring (Reranker)**
    *   The **Reranker Agent** (`agents/reranker_agent.py`) uses an LLM to read the *content* and score trials based on **severity** and **medical priority**.
    *   It promotes trials where the patient's condition is most urgent.

### 🛡️ Robustness
*   By embedding the `Study Status` and `Conditions` alongside the summary, the vector space naturally clusters "Active" trials for specific diseases together.
*   The system doesn't just find "text matches"—it finds **clinically relevant opportunities**.
