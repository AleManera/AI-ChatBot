from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login
from webscraper import BeautifulSoup
from webscraper import os

#login(token='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')

# Hugging Face API Token
hf_api_token = "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"

# Your fine-tuned model on Hugging Face
model_name = "AleManera/fine-tuned-llama"

# Initialize Hugging Face's Inference API
client = InferenceClient(model=model_name, token=hf_api_token)

# Define prompt template
template = """
Based on this text: {context}

Answer the question below:

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Abstracts and file mapping
abstracts = [
    "This is the homepage of Arol Group, which provides various industrial solutions including capping machines, bottle packaging, and more. Learn about their products, services, and history here.",
    "This page discusses Arol Group's customer care services, specifically for capping machines, offering support and solutions for clients in need of assistance.",
    "This page covers the latest news and events related to Arol Group, including product launches, industry updates, and participation in international trade shows.",
    "Arol Canelli is part of Arol Group and is located in Canelli, Italy. The company specializes in designing and manufacturing packaging solutions for the food and beverage industry.",
    "Arol Group is an international leader in the packaging industry, providing innovative solutions for bottling and capping processes. Discover their commitment to quality and sustainability.",
    "This page invites potential candidates to work with Arol Group, describing the company culture, career opportunities, and job openings within the organization.",
    "Arol Group provides contact information for clients, suppliers, and potential partners, facilitating communication through their official website."
]

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
    """Finds the most relevant document based on the input query."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(abstracts + [input_text])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    most_similar_idx = similarities.argmax()
    most_similar_score = similarities[0, most_similar_idx]

    if most_similar_score >= 0.3:
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
    """Handles the chatbot conversation loop."""
    print("Welcome to the Arol Group AI Chatbot! (Type exit to quit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        context = prepare_context(user_input)
        prompt_text = template.format(context=context, question=user_input)

        # Use Hugging Face Inference API
        response = client.text_generation(prompt_text, max_new_tokens=100)

        print(f"Bot: {response}")

if __name__ == "__main__":
    handle_conversation()