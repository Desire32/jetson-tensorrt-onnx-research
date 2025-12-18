from nano_llm import NanoLLM

# torch.cuda.empty_cache()

tested_models_mem_broken = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "stabilityai/stablelm-2-zephyr-1_6b",
]  # not enough memory to use without quantization
tested_models_broken = [
    "ibm-granite/granite-4.0-350m",
    "Qwen/Qwen3-0.6B",
    "google/gemma-3-270m",
]  # are not supported by any

working_slm_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "princeton-nlp/Sheared-LLaMA-1.3B",
]
working_tested_models_hf = [
    "HuggingFaceTB/SmolLM2-135M-Instruct"
]  # no quantization supported, bad

# AssertionError: Model type granitemoehybrid not supported.

# ValueError: The checkpoint you are trying to load has model type `granitemoehybrid` but Transformers does not recognize this architecture.
# This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

# ValueError: The checkpoint you are trying to load has model type `qwen3` but Transformers does not recognize this architecture.
# This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.


model = NanoLLM.from_pretrained(
    working_slm_models[
        0
    ],  # HuggingFace repo/model name, or path to HF model checkpoint
    api="hf",  # supported APIs are: mlc, awq, hf
    api_token="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",  # HuggingFace API key for authenticated models ($HUGGINGFACE_TOKEN)
    quantization="q4f16_1",  # q4f16_ft, q4f16_1, q8f16_0 for MLC, or path to AWQ weights
)

response = model.generate(
    "99 bottles of beer on the wall, 99 bottles of beer. Take one down and pass it around, 98 bottles of beer on the wall.",
    max_new_tokens=128,
)

for token in response:
    print(token, end="", flush=True)
