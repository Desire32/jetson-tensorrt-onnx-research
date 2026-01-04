# import os
# os.system("sync && echo 3 > /proc/sys/vm/drop_caches") # find a way to launch it with root permissions


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=5)

import termcolor
from nano_llm import NanoLLM

model = NanoLLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    api="mlc",
    api_token="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",
    quantization="q4f16_ft",
)

from nano_llm import ChatHistory

# create the chat history
chat_history = ChatHistory(
    model,
    system_prompt="You are a helpful and friendly AI assistant. DO NOT REPEAT YOURSELF",
    print_stats=True,
)


while True:
    # enter the user query from terminal
    print(">> ", end="", flush=True)
    prompt = input().strip()

    # add user prompt and generate chat tokens/embeddings
    chat_history.append("user", prompt)
    embedding, position = chat_history.embed_chat()

    # generate bot reply
    reply = model.generate(
        embedding,
        streaming=True,
        kv_cache=chat_history.kv_cache,
        stop_tokens=chat_history.template.stop,
        max_new_tokens=96,  # control the speech of a model
    )

    # stream the output
    for token in reply:
        termcolor.cprint(token, "blue", end="\n\n" if reply.eos else "", flush=True)

    # chat_history.append('bot', reply)
    nodes = retriever.retrieve(prompt)
    context = "\n\n".join([node.text for node in nodes])

    chat_history.append("user", f"Context: {context}\n\nQuestion: {prompt}")
