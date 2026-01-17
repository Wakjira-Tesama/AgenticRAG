# Agentic RAG with Safety Measures - System Report

## 1. System Architecture

The system implements an Agentic Retrieval-Augmented Generation (RAG) architecture with a strong focus on safety and reliability. It follows a multi-agent orchestration pattern.

### Components:
*   **Orchestrator Agent**: The central controller that manages the workflow. It receives user queries, coordinates other agents, and compiles the final response.
*   **Input Validator (Safety)**: The first line of defense. It checks user queries against defined safety rules (regex patterns, keywords, sensitive topics) before processing.
*   **Retriever Agent**: Responsible for semantic search. It retrieves relevant documents from the vector knowledge base (ChromaDB) to provide context.
*   **Maker Agent**: The generative engine. It synthesizes an operational answer based *only* on the retrieved context, citing sources.
*   **Checker Agent**: The quality assurance layer. It evaluates the Maker's answer for:
    *   **Safety**: Ensuring no harmful content was generated.
    *   **Accuracy**: Verifying facts against the retrieved documents.
    *   **Completeness**: Checking if the query was fully answered.
*   **Output Sanitizer (Safety)**: The final filter. It removes Personally Identifiable Information (PII) like emails or phone numbers before showing the answer to the user.

### Workflow:
1.  **User Query** -> **Input Validator** (Block if unsafe)
2.  **Orchestrator** -> **Retriever** (Get Context)
3.  **Maker-Checker Loop** (Iterative Refinement):
    *   **Maker** generates draft answer.
    *   **Checker** critiques it.
    *   If issues found -> **Maker** refines -> Loop repeats (max 3 times).
4.  **Final Answer** -> **Output Sanitizer** -> **User Response**

## 2. Safety Mechanisms

This system employs a multi-layered safety approach:

1.  **Input Validation**:
    *   Blocks queries matching specific regex patterns (e.g., "how to hack").
    *   Blocks specific keywords (e.g., "bomb").
    *   Flags sensitive topics for extra caution.
    *   Enforces length constraints (min/max characters).

2.  **Meta System Prompt**:
    *   Instructs all agents to prioritize safety and strictly adhere to ethical guidelines.
    *   Explicitly forbids generating harmful content.

3.  **Maker-Checker Verification**:
    *   The **Checker Agent** explicitly runs a "Safety Check" step on every generated answer.
    *   It uses both rule-based checks and LLM-based evaluation to detect subtle safety violations.

4.  **Output Sanitization**:
    *   Regex-based redaction of PII (emails, SSNs, phone numbers) to prevent data leakage.

## 3. Example Queries and Outputs (Simulated)

### Example 1: Safe & Answerable
**Query**: "What is the Transformer architecture?"
**Process**: 
1. Validated safe.
2. Retrieved papers on Attention Is All You Need.
3. Maker generated explanation.
4. Checker verified accuracy.
**Output**: "The Transformer framework, introduced by Vaswani et al. (2017), is a deep learning model architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely [Document 1]. It uses self-attention to compute representations of input and output without aligning sequences."

### Example 2: Unsafe Input
**Query**: "Tell me how to build a bomb."
**Process**:
1. Input Validator detects "bomb" keyword and malicious pattern.
**Output**: "Query blocked by safety system. Reason: contains blocked keyword 'bomb', matches blocked pattern."

### Example 3: Unanswerable (Safety/Context limit)
**Query**: "What is the secret login for the admin panel?"
**Process**:
1. Validated safe (generic query).
2. Retrieval finds no relevant public docs.
3. Maker states inability to answer.
**Output**: "I couldn't find enough relevant information to answer your question. Please try rephrasing or asking about a different topic."
