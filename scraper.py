# What format to save it in?
    # CSV?
    # Plain text?
    # one giant text file?
    # A different text file for each spell?
        # well i can make any of the other formats from this so I think I'll go with that

# Got some help from here: https://www.dataquest.io/blog/web-scraping-tutorial-python/
# and here: https://codeburst.io/the-ultimate-guide-to-web-scraping-in-python-3-7151425004c5

import requests
from bs4 import BeautifulSoup
# need to use this to activate javascript stuff
from selenium import webdriver
import pickle
import re
import time
from tqdm import tqdm
import os
from slugify import slugify

# Get the page with the list of links to spells
driver = webdriver.Chrome()
driver.get("https://www.dnd-spells.com/spells")

page_soup = BeautifulSoup(driver.page_source, 'html.parser')
# Okay so its the first a tag in each row in the tbody of the table with id 'example'
table = page_soup.find('table', id="example")
tbody = table.find('tbody')
rows = tbody.find_all('tr')
links = [row.find('a')['href'] for row in rows]

driver.quit()

pickle.dump(links, open("links_to_parse.pckl", "wb"))

links = pickle.load(open("links_to_parse.pckl", "rb"))

cnt = 0
spell_texts = {}
for link in tqdm(links):
    print(link)
    time.sleep(2) # Don't spam this site
    page = requests.get(link)
    page_soup = BeautifulSoup(page.content, 'html.parser')

    heading = page_soup.find("h1", attrs={"class":"classic-title"})
    cur_el = heading
    spell_name = cur_el.get_text()

    cur_el = cur_el.find_next_sibling("p")
    spell_school = cur_el.get_text()

    cur_el = cur_el.find_next_sibling("p")
    # Stack Overflow gave me a good regex to use to get rid of excessive spacing
    spell_stats = re.sub(r"\s{4,}","\n",cur_el.get_text().strip())

    cur_el = cur_el.find_next_sibling("p")
    spell_description = cur_el.get_text().strip()

    # Some spells have a description of what changes if you cast them at higher level
    if "At higher level" in cur_el.find_next_sibling("h4").get_text():
        cur_el = cur_el.find_next_sibling("p")
        spell_upcast_desc = "At higher levels: " + cur_el.get_text().strip()
    else:
        spell_upcast_desc = ""

    cur_el = cur_el.find_next_sibling("p")
    spell_book_location = cur_el.get_text().strip()

    cur_el = cur_el.find_next_sibling("p")
    # We gotta replace rediculous spacing again
    spell_classes = re.sub(r"\s{4,}"," ",cur_el.get_text().strip())
    spell_texts[spell_name] = (spell_name + "\n" + \
                            spell_school + "\n" + \
                            spell_stats + "\n" + \
                            spell_description + "\n" + \
                            spell_upcast_desc + "\n" + \
                            spell_book_location + "\n" + \
                            spell_classes).encode("ascii", "ignore")

pickle.dump(spell_texts, open("spell_texts.pckl", "wb"))

spell_texts = pickle.load(open("spell_texts.pckl", "rb"))

# Maybe I'm being too explicit here...
if not os.path.exists(os.path.join(os.getcwd(), "spells")):
    os.makedirs(os.path.join(os.getcwd(), "spells"))

for name, text in spell_texts.items():
    with open(os.path.join("spells", slugify(name)) + ".txt", "wb+") as fil:
        fil.write(text)
