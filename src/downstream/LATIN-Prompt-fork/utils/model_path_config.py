import os


model_path_config = {
    "blip2-opt-2.7b": "Salesforce/blip2-opt-2.7b",
    "blip2-opt-6.7b": "Salesforce/blip2-opt-6.7b",
    "opt-2.7b": "facebook/opt-2.7b",
    "opt-6.7b": "facebook/opt-6.7b",
    "llama-7b": "decapoda-research/llama-7b-hf",
    "alpaca-7b": os.path.expanduser("~/checkpoints/alpaca-7b"),
    "opt-iml-max-1.3b": "facebook/opt-iml-max-1.3b",
    "opt-iml-max-30b": "facebook/opt-iml-max-30b",
    "vicuna-13b": "lmsys/vicuna-13b-v1.3",
    # "llama2-7b-chat": "TheBloke/Llama-2-7B-chat-AWQ",
    # "llama2-13b-chat": "TheBloke/Llama-2-13B-chat-AWQ",
    "llama2-7b": "NousResearch/Llama-2-7b-hf",
    "llama2-7b-chat": "NousResearch/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "NousResearch/Llama-2-13b-chat-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    'zephyr-7b': 'HuggingFaceH4/zephyr-7b-alpha',
    # "meta-llama/Llama-2-13b-chat-hf",
}
