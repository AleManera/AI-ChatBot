from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from huggingface_hub import login
import json


# Load a pre-trained model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with the LLaMA model or any other language model you'd like to fine-tune
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Load the qa_pairs from the JSON file
with open('qa_pairs.json', 'r') as file:
    qa_pairs = json.load(file)

# Convert data to Dataset format
dataset = Dataset.from_dict({"question": [pair["question"] for pair in qa_pairs], 
                             "answer": [pair["answer"] for pair in qa_pairs]})

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["question"], examples["answer"], truncation=True, padding=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    evaluation_strategy="epoch",
    logging_dir='./logs',
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=10,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Ideally, use a separate validation set
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")