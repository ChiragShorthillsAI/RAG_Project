import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    """
    Handles text splitting into chunks using RecursiveCharacterTextSplitter.
    """
    def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""]
        )
    
    def split_text(self, text):
        """Splits text into chunks based on the configured text splitter."""
        return self.text_splitter.split_text(text)

class MovieTextProcessor:
    """
    Processes text files containing movie data, splits them into chunks,
    and saves the chunked data as a pickle file.
    """
    def __init__(self, folders, output_file="chunked_data.pkl"):
        self.folders = folders
        self.output_file = output_file
        self.chunker = TextChunker()
        self.chunked_data = {}
        self.total_chunk_count = 0
    
    def process_files(self):
        """Processes all text files in the given folders."""
        for folder in self.folders:
            self.process_folder(folder)
        
        print(f"\nTotal chunks generated: {self.total_chunk_count}")
        self.save_data()
    
    def process_folder(self, folder):
        """Processes all text files in a given folder."""
        folder_path = os.path.join(os.getcwd(), folder)
        self.chunked_data[folder] = {}
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                self.process_file(folder, folder_path, filename)
    
    def process_file(self, folder, folder_path, filename):
        """Processes a single text file, splits it into chunks, and stores the result."""
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            chunks = self.chunker.split_text(text)
            self.chunked_data[folder][filename] = chunks
            self.total_chunk_count += len(chunks)
            print(f"Processed {filename} in {folder} with {len(chunks)} chunks.")
    
    def save_data(self):
        """Saves the chunked data to a pickle file."""
        with open(self.output_file, "wb") as f:
            pickle.dump(self.chunked_data, f)
        print(f"Chunked data saved to '{self.output_file}'.")

if __name__ == "__main__":
    folders = ["movies_2019", "movies_2020", "movies_2021", "movies_2022", "movies_2023"]
    processor = MovieTextProcessor(folders)
    processor.process_files()
