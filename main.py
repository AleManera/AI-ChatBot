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
import gradio as gr
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

Don't mention the text you based your answer on. Or at most, refer to it as the company databases.
"""


#model_path = "./fine_tuned_model" 
# Load the fine-tuned model
model = OllamaLLM(model="llama3")
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

#model_name = "AleManera/fine-tuned-llama"

# Load the model from Hugging Face Hub (fine-tuned model)
#model = AutoModelForCausalLM.from_pretrained(model_name)
#hf_api_token = "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"
#tokenizer = AutoTokenizer.from_pretrained(model_name)

#pipe = pipeline("text-generation", model=model, token=hf_api_token)
#llm = HuggingFacePipeline(pipeline=pipe)

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

abstracts = [
    "This is the homepage of Arol Group, which provides various industrial solutions including capping machines, bottle packaging, and more. Learn about their products, services, and history here.",
    "This page discusses Arol Group's customer care services, specifically for capping machines, offering support and solutions for clients in need of assistance.",
    "This page covers the latest news and events related to Arol Group, including product launches, industry updates, and participation in international trade shows. Additionally exhibitions, shows and any general news.",
    "Arol Canelli is part of Arol Group and is located in Canelli, Italy. The company specializes in designing and manufacturing packaging solutions for the food and beverage industry.",
    "Arol Group is an international leader in the packaging industry, providing innovative solutions for bottling and capping processes. Discover their commitment to quality and sustainability.",
    "This page invites potential candidates to work with Arol Group, describing the company culture, career opportunities, and job openings within the organization.",
    "Arol Group provides contact information for clients, suppliers, and potential partners, facilitating communication through their official website."
]

# Mapping the abstracts to corresponding file paths (representing each URL)
myDict = {
    abstracts[0]: "www.arol.com_.txt", #home
    abstracts[1]: "www.arol.com_customer-care-for-capping-machines.txt", #customer_care
    abstracts[2]: "www.arol.com_news-events.txt", #news_events
    abstracts[3]: "www.arol.com_arol-canelli.txt", #company
    abstracts[4]: "www.arol.com_arol-group-canelli.txt", #arol_group
    abstracts[5]: "www.arol.com_work-with-us.txt", #work_with_us
    abstracts[6]: "www.arol.com_arol-contact.txt" #contacts
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

def prepare_context_merged_file():
    with open("merged_cleaned.txt", "r", encoding="utf-8") as f:
        return f.read()


def prepare_context(input_text):
    # Compute similarity between input_text and abstracts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(abstracts + [input_text])  # Include user input in the matrix
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])  # Compare input_text with abstracts
    most_similar_idx = similarities.argmax()  # Get index of the most similar abstract
    most_similar_score = similarities[0, most_similar_idx]

    print(f"Matched abstract: {abstracts[most_similar_idx]}")
    print(f"Similarity score: {most_similar_score}")

    if most_similar_score >= 0.01:  # Threshold
        matched_abstract = abstracts[most_similar_idx]
        file_path = myDict.get(matched_abstract)
        print(f"Most similar index: {most_similar_idx}")
        print(f"File path: {file_path}")
        if file_path and os.path.exists(file_path):
            with open(file_path, "r") as f:
                return f.read()
        else:
            return f"Error: The file for '{matched_abstract}' was not found."
    else:
        return "I'm sorry, I couldn't find relevant information for your query."
    

def summarize_conversation(conversation_history):
    """
    Summarizes old conversation history to retain relevant context while reducing length.
    """
    summary_prompt = f"""
    Summarize the following conversation while keeping the key details relevant to future responses:

    {conversation_history}

    Summary:
    """

    summary = chain.invoke({"context": "", "question": summary_prompt})  # Use chatbot model to summarize
    return summary


def handle_conversation(input_text):
    #conversation_history = load_chat_history()
    conversation_history = ""
    #recent_messages = []  # Stores the last 10 interactions

    print("Welcome to the Arol Group AI Chatbot! (Type 'exit' to quit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            #save_chat_history(conversation_history)  # Save history when exiting
            break

        context = prepare_context_merged_file()
        full_context = conversation_history + f"\nUser: {user_input}\n"

        print("Bot: ")
        result = chain.invoke({"context": context + full_context, "question": user_input})

        # Store recent messages
        conversation_history = conversation_history + f"\nUser: {user_input}\nBot: {result}"

        # If too many messages, summarize old conversations
        #if len(recent_messages) > 10:
        #    summarized_context = summarize_conversation(conversation_history)
        #    conversation_history = summarized_context + "\n" + "\n".join(recent_messages[-5:])  # Keep last 5 detailed messages
        #    recent_messages = recent_messages[-5:]  # Retain last 5 detailed messages only

        print(result)

def handle_conversation_gradio(user_input, conversation_history=""):
    if user_input.lower() == "exit":
        return "Goodbye!"

    # Prepare context
    context = prepare_context_merged_file()  # Make sure this function is defined
    full_context = conversation_history + f"\nUser: {user_input}\n"

    # Get bot response
    result = chain.invoke({"context": context + full_context, "question": user_input})

    # Update conversation history
    conversation_history += f"\nUser: {user_input}\nBot: {result}"

    return result  # Return both response and updated history

iface = gr.Interface(
            fn=handle_conversation_gradio,
            inputs="text",
            outputs="text",
            title="Arol Group AI Chatbot",
            description="Ask me anything about Arol Group!"
        )


if __name__ == "__main__":
    #handle_conversation()
    iface.launch()