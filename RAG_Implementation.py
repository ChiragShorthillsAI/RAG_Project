import os
import numpy as np
import pickle
import unicodedata
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class PineconeIndexManager:
    """Handles Pinecone index initialization and connection."""
    def __init__(self, index_name="rag", dimension=768):
        load_dotenv()
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)

class EmbeddingGenerator:
    """Generates embeddings using Google Generative AI."""
    def __init__(self, model_name="models/embedding-001"):
        self.model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    def embed_text(self, text):
        """Generates an embedding for a given text chunk."""
        return self.model.embed_query(text)

class TextProcessor:
    """Loads text data, splits it into chunks, and prepares it for embedding."""
    def __init__(self, base_path, folders, chunk_size=2000, chunk_overlap=200):
        self.base_path = base_path
        self.folders = folders
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def load_text_files(self):
        """Loads text files from specified folders."""
        all_documents = []
        for folder in self.folders:
            folder_path = os.path.join(self.base_path, folder)
            if os.path.exists(folder_path):
                loader = DirectoryLoader(folder_path, glob="*.txt", loader_cls=TextLoader)
                docs = loader.load()
                all_documents.extend(docs)
            else:
                print(f"Warning: folder not found -> {folder_path}")
        return all_documents
    
    def split_text(self, documents):
        """Splits text into manageable chunks."""
        return self.text_splitter.split_documents(documents)

class RAGSystem:
    """Implements a Retrieval-Augmented Generation (RAG) system for answering questions."""
    def __init__(self, retriever, model="gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the retrieved context to answer the question. {context}"),
            ("human", "{input}"),
        ])
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(retriever, self.question_answer_chain)
    
    def ask_question(self, question):
        """Retrieves relevant documents and generates an answer."""
        return self.rag_chain.invoke({"input": question})


base_path = "/home/shtlp_0132/Desktop/Work/RAG_Project"
folders = ["movies_2019", "movies_2020", "movies_2021", "movies_2022", "movies_2023"]

pinecone_manager = PineconeIndexManager()
embedding_generator = EmbeddingGenerator()
text_processor = TextProcessor(base_path, folders)

docsearch = PineconeVectorStore.from_existing_index(index_name=pinecone_manager.index_name, embedding=embedding_generator.model)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

rag_system = RAGSystem(retriever)

if __name__ == "__main__":
    print("Ready to answer questions!")
