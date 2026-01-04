# README.md — Model Comparison from Screenshots

| Parameter                 | Sheared-LLaMA-1.3B             | TinyLlama-1.1B-Chat-v1.0         | Winner           |
|---------------------------|--------------------------------|----------------------------------|------------------|
| Actual parameters         | ~1.3B                          | ~1.1B                            | TinyLlama        |
| Quant size (Q4f16)        | 722–651 MB                     | 533–590 MB                       | TinyLlama        |
| Layers                    | 24                             | 22                               | TinyLlama (lighter) |
| Attention heads           | 16 (KV 16)                     | 32 (KV 4, GQA)                   | TinyLlama (efficient) |
| Context length            | 4096                           | 2048                             | Sheared          |
| RoPE theta                | none                           | 10000                            | TinyLlama        |
| Data type                 | float32                        | bfloat16                         | TinyLlama (precise) |
| Optimization              | base                           | chat (DPO)                       | TinyLlama        |
| Load time                 | 7–8 sec                        | 5–7 sec                          | TinyLlama        |
| Generation quality (beer song) | repeats, degradation       | coherent, meaningful             | TinyLlama        |

## Winner: **TinyLlama-1.1B-Chat-v1.0**

Best for chat tasks and local inference on all key metrics except context length.
