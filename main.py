from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from webscraper import BeautifulSoup   
import os

template = """
Based on this text: {context}

Answer the question below:

Question: {question}

Answer:
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

abstracts = ["home page","this talks about football","this talks about cristiano ronaldo"]
myDict = {"home page":"home_page.html",
          "cristiano ronaldo":"this talks about football"}


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
    
    #ret = "Il mio amore è molto bello me lo mangio col coltello tutta notte abbracciati come veri innamorati"
    #ret = f.read(myDict[abstracts[0]],'r').toText()
    #return ret

def handle_conversation():
    context = ""

    print("Welcome to the Arol Group AI Chatbot! (Type exit to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        context = prepare_context(user_input)
        print("Bot: ")
        result = chain.invoke({"context": context, "question": user_input}, stream=True)
        for chunk in result:
            print(chunk, end="", flush=True)
        print()
        context += f"\nUser: {user_input}\nBot: {result}"
        
if __name__ == "__main__":
    handle_conversation()