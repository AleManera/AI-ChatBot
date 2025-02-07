import os
import re

# Directory where your text files are stored
input_files = ["www.arol.com_.txt", "www.arol.com_customer-care-for-capping-machines.txt", "www.arol.com_news-events.txt", "www.arol.com_arol-canelli.txt", "www.arol.com_arol-group-canelli.txt", "www.arol.com_work-with-us.txt", "www.arol.com_arol-contact.txt"]
output_file = "merged_cleaned.txt"

# Function to clean text
def clean_text(text):
    # Remove excessive blank lines and spaces
    text = re.sub(r"\n\s*\n", "\n", text)  # Remove multiple blank lines
    text = re.sub(r"\s{2,}", " ", text)  # Replace multiple spaces with a single space
    
    # Remove typical website navigation elements
    navigation_keywords = [
        "Home", "Sectors", "Customer care", "News & Events", "Company",
        "Arol Group", "Work with us", "Contacts", "Terms & Conditions",
        "Privacy Policy", "Cookie Policy", "Legal Notes", "Code of Ethics"
    ]
    
    # Remove lines that contain navigation words
    text = "\n".join(
        line for line in text.split("\n") if not any(nav in line for nav in navigation_keywords)
    )
    
    # Strip leading/trailing spaces from each line
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

    return text

# Merge all files
merged_text = ""

for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
        cleaned_content = clean_text(content)
        merged_text += cleaned_content + "\n\n"  # Add spacing between files

# Save cleaned merged text
with open(output_file, "w", encoding="utf-8") as out_file:
    out_file.write(merged_text)
