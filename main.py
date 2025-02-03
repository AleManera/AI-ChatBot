from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics.pairwise import cosine_similarity
from webscraper import BeautifulSoup   
from webscraper import os
import torch
from huggingface_hub import HfApi, HfFolder

api = HfApi()

# repo_name = "fine-tuned-llama" 
# repo_id = "AleManera/fine-tuned-llama" 
# api.create_repo(repo_id, repo_type="model")

# api.upload_folder(
#     folder_path="./fine_tuned_model",
#     repo_id="AleManera/fine-tuned-llama",
#     repo_type="model"
# )


template = """
Based on this text: {context}

Answer the question below:

Question: {question}

Answer:
"""


#model_path = "./fine_tuned_model" 
# Load the fine-tuned model
#model = OllamaLLM(model=model_path)
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModelForCausalLM.from_pretrained(model_path)
# Load the base model
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# Use a quantized version to save memory (4-bit quantization)
#model = AutoModelForCausalLM.from_pretrained(
#    "meta-llama/Llama-2-7b-chat-hf", 
#    torch_dtype=torch.float16,  # Use FP16 instead
#    device_map="mps"  # For Apple Silicon Macs
#)

# Load the LoRA fine-tuned model with the adapter
#adapter_model = PeftModel.from_pretrained(model, "./fine_tuned_model")

# Load the tokenizer
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

model_name = "AleManera/fine-tuned-llama"

# Load the model from Hugging Face Hub (fine-tuned model)
model = AutoModelForCausalLM.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model, token=hf_api_token)
llm = HuggingFacePipeline(pipeline=pipe)

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

abstracts = [
    "This is the homepage of Arol Group, which provides various industrial solutions including capping machines, bottle packaging, and more. Learn about their products, services, and history here.",
    "This page discusses Arol Group's customer care services, specifically for capping machines, offering support and solutions for clients in need of assistance.",
    "This page covers the latest news and events related to Arol Group, including product launches, industry updates, and participation in international trade shows.",
    "Arol Canelli is part of Arol Group and is located in Canelli, Italy. The company specializes in designing and manufacturing packaging solutions for the food and beverage industry.",
    "Arol Group is an international leader in the packaging industry, providing innovative solutions for bottling and capping processes. Discover their commitment to quality and sustainability.",
    "This page invites potential candidates to work with Arol Group, describing the company culture, career opportunities, and job openings within the organization.",
    "Arol Group provides contact information for clients, suppliers, and potential partners, facilitating communication through their official website."
]

# Mapping the abstracts to corresponding file paths (representing each URL)
myDict = {
    "home": "www.arol.com_home.txt",
    "customer_care": "www.arol.com_customer_care_for_capping_machines.txt",
    "news_events": "www.arol.com_news_events.txt",
    "company": "www.arol.com_arol_canelli.txt",
    "arol_group": "www.arol.com_arol_group_canelli.txt",
    "work_with_us": "www.arol.com_work_with_us.txt",
    "contacts": "www.arol.com_arol_contact.txt"
}

def html_to_text(html_file):
    """Converts an HTML file to plain text."""
    with open(html_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse the HTML
    soup = BeautifulSoup(content, 'html.parser')

    # Extract plain text
    text = soup.get_text(separator="\n")  # Use '\n' to preserve structure for paragraphs

    return text

def prepare_context(input_text):
    # Compute similarity between input_text and abstracts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(abstracts + [input_text])  # Include user input in the matrix
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Compare input_text with abstracts
    most_similar_idx = similarities.argmax()  # Get index of the most similar abstract
    most_similar_score = similarities[0, most_similar_idx]

    if most_similar_score >= 0.3:  # Threshold
        matched_abstract = abstracts[most_similar_idx]
        file_path = myDict.get(matched_abstract)
        if file_path and os.path.exists(file_path):
            with open(file_path, "r") as f:
                return f.read()
        else:
            return f"Error: The file for '{matched_abstract}' was not found."
    else:
        return "I'm sorry, I couldn't find relevant information for your query."

def handle_conversation():
    context = ""

    print("Welcome to the Arol Group AI Chatbot! (Type exit to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        context = prepare_context(user_input)
        print("Bot: ")
        result = chain.invoke({"context": context, "question": user_input})#, stream=True)
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
        context += f"\nUser: {user_input}\nBot: {result}"

if __name__ == "__main__":
    handle_conversation()