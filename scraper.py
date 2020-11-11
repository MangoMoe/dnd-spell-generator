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

# # Get the page with the list of links to spells
# driver = webdriver.Chrome()
# driver.get("https://www.dnd-spells.com/spells")

# page_soup = BeautifulSoup(driver.page_source, 'html.parser')
# # Okay so its the first a tag in each row in the tbody of the table with id 'example'
# table = page_soup.find('table', id="example")
# tbody = table.find('tbody')
# rows = tbody.find_all('tr')
# links = [row.find('a')['href'] for row in rows]

# # for i in range(100):
# #     print(links[i])

# driver.quit()

# pickle.dump(links, open("links_to_parse.pckl", "wb"))

links = pickle.load(open("links_to_parse.pckl", "rb"))

# TODO make folders and save results to text files

cnt = 0
for link in tqdm(links):
    print("\n\n\n")
    time.sleep(2) # Don't spam this site
    page = requests.get(link)
    page_soup = BeautifulSoup(page.content, 'html.parser')
    # sometimes the "at higher level" section is there, sometimes it is not
    # use .find(string='blah') to find text in an element
    # well it looks like i'll have to manually code each part it should take in...

    heading = page_soup.find("h1", attrs={"class":"classic-title"})
    cur_el = heading
    # spell_name = cur_el.find("span").string
    spell_name = cur_el.get_text()
    print(spell_name)

    # spell_school = spell_name_span.next_element.string
    cur_el = cur_el.find_next_sibling("p")
    spell_school = cur_el.get_text()
    print(spell_school)

    cur_el = cur_el.find_next_sibling("p")
    # SO gave me a good regex to use to get rid of excessive spacing
    spell_stats = re.sub(r"\s{4,}","\n",cur_el.get_text().strip())
    print(spell_stats)

    cur_el = cur_el.find_next_sibling("p")
    spell_description = cur_el.get_text().strip()
    print(spell_description)

    # Some spells have a description of what changes if you cast them at higher level
    if "At higher level" in cur_el.find_next_sibling("h4").get_text():
        cur_el = cur_el.find_next_sibling("p")
        spell_upcast_desc = "At higher levels: " + cur_el.get_text().strip()
    else:
        spell_upcast_desc = ""
    print(spell_upcast_desc)

    cur_el = cur_el.find_next_sibling("p")
    spell_book_location = cur_el.get_text().strip()
    print(spell_book_location)

    cur_el = cur_el.find_next_sibling("p")
    # We gotta replace rediculous spacing again
    spell_classes = re.sub(r"\s{4,}"," ",cur_el.get_text().strip())
    print(spell_classes)

    if cnt > 10:
        break
    cnt += 1