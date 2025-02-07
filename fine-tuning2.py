from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import os
os.environ["WANDB_DISABLED"] = "true"

# Load dataset from JSON
dataset = load_dataset("json", data_files="qa_pairs.json")  # Ensure JSON format is valid

# Load tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Function to tokenize input
def tokenize_function(examples):
    inputs = tokenizer(examples["question"], examples["answer"],
                       truncation=True, padding="max_length", max_length=512)

    # Set labels to be the same as input_ids
    inputs["labels"] = inputs["input_ids"].copy()

    return inputs

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load model in LoRA mode (for memory-efficient fine-tuning)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

# Apply LoRA (parameter-efficient tuning)
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Modify only attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=2,  # Adjust based on GPU memory
    gradient_accumulation_steps=16,  # Simulates larger batch size
    num_train_epochs=10,
    evaluation_strategy="no",
    logging_dir="./logs",
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")