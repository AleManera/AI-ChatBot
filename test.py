from huggingface_hub import InferenceClient
from transformers import pipeline

hf_api_token = "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"
model_name = "AleManera/fine-tuned-llama"
client = InferenceClient(model=model_name, token=hf_api_token)

# Manually specify the pipeline
inference = pipeline("text-generation", model=model_name, tokenizer=model_name)

# Now use the pipeline to generate text
response = inference("How are you today?")
print(response)
