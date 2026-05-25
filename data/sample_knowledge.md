# ContextLens AI Knowledge Base

Retrieval-Augmented Generation, usually called RAG, is a technique for improving large language model answers by adding relevant external knowledge before the model generates a response.

Instead of asking an LLM to answer only from its internal training data, a RAG system first searches a knowledge base. The most relevant passages are inserted into the model prompt as context. This makes the answer more grounded, more current, and easier to verify.

## Core Components

A complete RAG workflow includes document loading, text chunking, embedding generation, vector storage, semantic retrieval, prompt construction, and LLM generation.

Document loading brings source files into the system. Text chunking splits long documents into smaller passages. Embedding models convert each passage into a numerical vector. A vector database stores these vectors and supports similarity search. The retriever finds passages close to the user's question. The LLM uses those passages to produce a final answer.

## Why RAG Reduces Hallucinations

LLMs can hallucinate when they rely only on learned patterns instead of source material. RAG reduces this risk by giving the model retrieved evidence and instructing it to answer only from that evidence. If the retrieved context does not contain the answer, the system should say it does not know.

## Production Considerations

Production RAG systems need document refresh pipelines, monitoring, permissions, evaluation datasets, and fallback behavior. Teams should measure retrieval quality, answer accuracy, latency, and citation reliability.

For sensitive data, production systems should enforce access control before retrieval so users only see content they are allowed to access.
