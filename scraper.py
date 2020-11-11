# What format to save it in?
    # CSV?
    # Plain text?
    # one giant text file?
    # A different text file for each spell?
        # well i can make any of the other formats from this so I think I'll go with that

# Got some help from here: https://www.dataquest.io/blog/web-scraping-tutorial-python/

import requests
from bs4 import BeautifulSoup
# need to use this to activate javascript stuff
from selenium import webdriver

# Get the page with the list of links to spells
driver = webdriver.Chrome()
driver.get("https://www.dnd-spells.com/spells")
# page = requests.get("https://www.dnd-spells.com/spells")

# For some reason, the scraped page and the one on the browser are different,
    # Not sure how they did it, but I think it is filtering out proprietary names
    # It looks like the scrape might be only getting things from the player's handbook
    # Yup that was it, adding selenium managed to get all the things
with open("test.html", "w") as fil:
    # fil.write(page.content)
    fil.write(driver.page_source)
# print(page.status_code)

# page_soup = BeautifulSoup(page.content, 'html.parser')
page_soup = BeautifulSoup(driver.page_source, 'html.parser')
# Okay so its the first a tag in each row in the tbody of the table with id 'example'
table = page_soup.find('table', id="example")
tbody = table.find('tbody')
rows = tbody.find_all('tr')
links = [row.find('a')['href'] for row in rows]

for i in range(10):
    print(links[i])

driver.quit()

# for link in links:
#     page = requests.get(link)
#     page_soup = BeautifulSoup(page.content, 'html.parser')
#     # sometimes the "at higher level" section is there, sometimes it is not
#     # use .find(string='blah') to find text in an element
#     # well it looks like i'll have to manually code each part it should take in...