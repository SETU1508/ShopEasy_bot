# ShopEasy_bot
# ShopEasy FAQ Bot — Agentic AI Capstone

An intelligent customer support chatbot for a fictional e-commerce store.

## Features

- **RAG Retrieval** — Answers from a 10-document FAQ knowledge base using ChromaDB
- **Multi-turn Memory** — Remembers your name and conversation context across turns
- **Price Calculator** — Calculates discounts and totals on request
- **Smart Routing** — Automatically routes questions to the right pipeline
- **Self-Evaluation** — Scores answer faithfulness and retries if quality is low
- **Grounded Answers** — Never fabricates; admits when it doesn't know

## Architecture — 8-Node LangGraph Pipeline

memory_node → router_node → [retrieval_node / tool_node / skip_node]
→ answer_node → eval_node → save_node → END

## 🛠️ Tech Stack

| Component       | Technology                           |
| --------------- | ------------------------------------ |
| LLM             | Groq (LLaMA3-8b-8192)                |
| Agent Framework | LangGraph StateGraph                 |
| Vector Database | ChromaDB (in-memory)                 |
| Embeddings      | SentenceTransformer all-MiniLM-L6-v2 |
| Memory          | LangGraph MemorySaver                |
| Evaluation      | RAGAS (faithfulness metric)          |
| UI              | Streamlit                            |
# E-Commerce-FAQ-bot
ShopEasy FAQ Bot — Agentic AI capstone project built with LangGraph, ChromaDB, Groq (LLaMA3), and Streamlit. Features RAG-based FAQ retrieval, multi-turn memory, discount calculator, and RAGAS evaluation.
