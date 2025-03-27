import os
import numpy as np # type: ignore
import pickle
import unicodedata
from langchain_groq import ChatGroq
from dotenv import load_dotenv # type: ignore
from pinecone import Pinecone # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_pinecone import PineconeVectorStore # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.document_loaders import DirectoryLoader, TextLoader # type: ignore
from langchain.chains.retrieval import create_retrieval_chain # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore

class PineconeIndexManager:
    """Handles Pinecone index initialization and connection."""
    def __init__(self, index_name="rag-final", dimension=768):
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
    def __init__(self, base_path, folders, chunk_size=1000, chunk_overlap=200):
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
    def __init__(self, retriever, model="llama-3.1-8b-instant"):
        # self.llm = ChatGoogleGenerativeAI(model=model)
        self.llm = ChatGroq(model=model,groq_api_key=os.getenv("GROQ_API_KEY"))
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise, context-based question-answering assistant. Follow these rules:Use Only Provided Context:Respond strictly based on the retrieved context. Do not add, infer, or assume anything not explicitly stated.Be Concise and Direct:Give point-to-point, factual answers without elaboration or repetition.Follow Instructions Exactly:If context is unclear, reply: Insufficient context. Please provide more information. If context is missing reply: I am sorry, but the provided context does not contain information. Maintain Accuracy and Tone:Ensure all responses are accurate, professional, and neutral—no opinions or speculation.Your Answer should be in at max 2 lines."),
            ("human", "Context: {context}\n\nQuestion: {input}"),
        ])
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(retriever, self.question_answer_chain)
    
    def ask_question(self, question):
        """Retrieves relevant documents and generates an answer, including retrieved chunks."""
        result = self.rag_chain.invoke({"input": question})
        answer = result.get("answer", "No answer found.")
        
        # Extract the retrieved documents from intermediate steps
        retrieved_docs = result.get("context", [])  # 'context' holds retrieved docs from retriever
        chunks = [doc.page_content for doc in retrieved_docs]

        return {
            "answer": answer,
            "chunks": chunks
        }


base_path = "/home/shtlp_0132/Desktop/Work/RAG_LLM_Project/"
folders = ["Movies_2019", "Movies_2020"]

# Initialize the necessary components
pinecone_manager = PineconeIndexManager()
embedding_generator = EmbeddingGenerator()
text_processor = TextProcessor(base_path, folders)

# Load and split the documents
documents = text_processor.load_text_files()  # Load text files from the folders
split_documents = text_processor.split_text(documents)  # Split the loaded documents into chunks

# Create the Pinecone Vector Store
docsearch = PineconeVectorStore.from_documents(
    documents=split_documents,  # Use the split documents (chunks) here
    index_name=pinecone_manager.index_name,
    embedding=embedding_generator.model
)

# docsearch = PineconeVectorStore.from_existing_index(index_name=pinecone_manager.index_name, embedding=embedding_generator.model)

# Create the retriever for querying
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the RAG System
rag_system = RAGSystem(retriever)

if __name__ == "__main__":
    print("Ready to answer questions!")

