| LLM (Ollama)                  | Embedding model                      | LLM Params | Release Year | RAG Answer Quality (1–10) | Notes                              |
|-------------------------------|--------------------------------------|------------|--------------|---------------------------|------------------------------------|
| Qwen3-0.6b                    | google/embedding-gemma-300m          | 600M       | 2025         | 9.5                       | Best overall balance               |
| Qwen3-0.6b                    | mixedbread-ai/mxbai-embed-large-v1   | 600M       | 2025         | 9.0                       | Excellent                          |
| Granite-4.0-h-350m            | google/embedding-gemma-300m          | 340M       | 2024         | 7.0                       | Accurate but dry                   |
| Granite-4.0-h-350m            | mixedbread-ai/mxbai-embed-large-v1   | 340M       | 2024         | 6.5                       | Acceptable                         |
| Qwen3-0.6b                    | sentence-transformers/all-MiniLM-L6-v2 | 600M     | 2025         | 7.5                       | Good LLM saves weaker embeddings   |
| Granite-4.0-h-350m            | sentence-transformers/all-MiniLM-L6-v2 | 340M     | 2024         | 5.5                       | Weak retrieval hurts               |
| SmolLM2-360M-Instruct         | google/embedding-gemma-300m          | 360M       | 2024         | 2.0                       | Heavy hallucinations               |
| SmolLM2-135M                  | google/embedding-gemma-300m          | 135M       | 2024         | 1.0                       | Complete disaster                  |

**Winner for lightweight RAG on Colab (T4 15 GB):**
Qwen3-0.6b + google/embedding-gemma-300m  
→ Near 7B-level quality at only ~3 GB VRAM.
