import os
import re
import requests
from bs4 import BeautifulSoup

class WebScraper:
    """
    A generic web scraper class with helper functions for fetching and parsing web pages.
    """
    @staticmethod
    def fetch_page(url):
        """Fetches a web page and returns its BeautifulSoup object."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching page {url}: {e}")
            return None
    
    @staticmethod
    def sanitize_filename(name):
        """Removes invalid characters from filenames."""
        return re.sub(r'[\\/*?:"<>|]', "", name)

class WikipediaMovieScraper(WebScraper):
    """
    Scrapes movie data from Wikipedia's list of films and saves it into text files.
    """
    BASE_URL = "https://en.wikipedia.org"

    def __init__(self, url, output_folder="movies_2019"):
        self.url = url
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
    def extract_linked_page_text(self, url):
        """Fetches linked Wikipedia page text."""
        soup = self.fetch_page(url)
        if soup:
            content_div = soup.find('div', class_='mw-parser-output')
            if content_div:
                paragraphs = content_div.find_all('p')
                return "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            return soup.get_text(strip=True)
        return ""
    
    def scrape_and_save_movies(self):
        """Scrapes the Wikipedia page and saves movie details."""
        soup = self.fetch_page(self.url)
        if not soup:
            return
        
        tables = soup.find_all('table', class_='wikitable')
        for table in tables:
            self.process_table(table)
    
    def process_table(self, table):
        """Processes a table of movie data."""
        rows = table.find_all('tr')
        for row in rows:
            data_cells = row.find_all('td')
            if not data_cells:
                continue
            
            movie_title = data_cells[0].get_text(strip=True)
            file_name = self.sanitize_filename(movie_title)
            if not file_name:
                continue
            file_path = os.path.join(self.output_folder, f"{file_name}.txt")
            
            row_text = "\n".join(cell.get_text(strip=True) for cell in data_cells)
            additional_texts = self.extract_linked_pages(data_cells)
            
            full_text = row_text + "\n" + "\n".join(additional_texts)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            print(f"Saved movie: {movie_title} to {file_path}")
    
    def extract_linked_pages(self, data_cells):
        """Extracts text from linked Wikipedia pages."""
        additional_texts = []
        for cell in data_cells:
            anchors = cell.find_all('a', href=True)
            for anchor in anchors:
                href = anchor['href']
                if not href.startswith('http'):
                    href = self.BASE_URL + href
                link_text = anchor.get_text(strip=True)
                linked_page_text = self.extract_linked_page_text(href)
                additional_texts.append(f"\nLinked page for '{link_text}' ({href}):\n{linked_page_text}")
        return additional_texts

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/List_of_American_films_of_2019"
    scraper = WikipediaMovieScraper(url)
    scraper.scrape_and_save_movies()
