import requests
from bs4 import BeautifulSoup

url_home = 'https://www.arol.com/'
url_customer_care = 'https://www.arol.com/customer-care-for-capping-machines'
url_news_events = 'https://www.arol.com/news-events'
url_company = 'https://www.arol.com/arol-canelli'
url_arol_group = 'https://www.arol.com/arol-group-canelli'
url_work_with_us = 'https://www.arol.com/work-with-us' 
url_contacts = 'https://www.arol.com/arol-contact'

urls = [url_home, url_customer_care, url_news_events, url_company, url_arol_group, url_work_with_us, url_contacts]

for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.prettify()
    text = content.get_text(separator="\n")

    filename = f'{url}.txt'
    with open(filename, 'w') as file:
        file.write(text)
