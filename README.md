### Commands

- `--model` — HuggingFace repo/model name, or path to HF model checkpoint
  The task was done using:
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - `princeton-nlp/Sheared-LLaMA-1.3B`

- `--quantization` — type of quantization  
  - `q0f32`, `q0f16` Plain model
  - `q3f16_0`, `q3f16_1` INT3
  - `q4f16_0`, `q4f16_1`, `q4f16_ft` INT4
  - `q8f16_ft` INT8

- `--max_new_tokens` — max number of new tokens

- `--embed` — embed models
  Was tested with:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `sentence-transformers/all-MiniLM-L12-v2`

- `--api` — backend / API  
  Возможные значения:
  - `mlc` main one to use
  - `awq` needs to have last version of JetsonPack
  - `hf`  base version without any modifications

- `--test-mode` — debug purposes with predefined templates

- `--chunk_size` — dividing text on chunks

- `--chunk_overlap` — overlap between

- `--top_k` — nearest doc pos for retrieval


## Problems to Solve

- `ValueError` caused by a critical security vulnerability in `torch.load`
  PyTorch version requirement: `torch >= 2.6` (does not apply to `safetensors`)
  Related vulnerability: CVE-2025-32434  
  https://nvd.nist.gov/vuln/detail/CVE-2025-32434

- Sweep objective: maximize `tokens_per_second` under limited memory

- Key hyperparameters:
  - KV-cache configuration
  - TensorRT / ONNX execution parameters

- Techniques to further optimize (PROBABLY):
  - Huffman / entropy coding
  - Sparse weights
  - Weight pruning (`torch.nn.utils.prune`)
