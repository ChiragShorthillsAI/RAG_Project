import os
import json
import time
import csv
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Pinecone and Langchain related imports
from pinecone import ServerlessSpec, Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
import google.generativeai as genai

# RAGAS related imports
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    SemanticSimilarity,
    AnswerCorrectness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configure logging
LOG_FILE = "process.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)


class PineconeHandler:
    """
    Handles connection to Pinecone and creates a retriever using a given index.
    """
    def __init__(self, api_key: str, index_name: str, cloud: str = "aws", region: str = "us-east-1"):
        self.api_key = api_key
        self.index_name = index_name
        self.spec = ServerlessSpec(cloud=cloud, region=region)
        # Initialize Pinecone instance
        self.pc = Pinecone(api_key=self.api_key)
        # Create embeddings instance
        self.embedding_instance = self.create_embeddings()
        # Build Pinecone vector store and retriever
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embedding_instance
        )
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        logging.info("PineconeHandler initialized.")

    def create_embeddings(self):
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )


class DataLoader:
    """
    Loads data from various sources (CSV, JSONL) and provides methods for data retrieval.
    """
    def __init__(self, csv_path: str = None, jsonl_path: str = None):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path

    @staticmethod
    def traverse_jsonl(file_path: str):
        """Reads a JSONL file and extracts user inputs and references."""
        ui_list = []
        ref_list = []

        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return ui_list, ref_list

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for idx, line in enumerate(file, start=1):
                    try:
                        obj = json.loads(line.strip())
                        user_input = obj.get("user_input", "N/A")
                        reference = obj.get("reference", "N/A")
                        ui_list.append(user_input)
                        ref_list.append(reference)
                        logging.info(f"Processed line {idx}: User Input = {user_input[:50]}...")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error at line {idx}: {str(e)}")
            logging.info(f"Completed processing JSONL file: {file_path}")
        except Exception as e:
            logging.error(f"Unexpected error while processing {file_path}: {str(e)}")
        return ui_list, ref_list

    def load_csv_data(self):
        """Loads CSV data and returns lists of questions, references, and answers."""
        if self.csv_path is None or not os.path.exists(self.csv_path):
            logging.error(f"CSV file not found: {self.csv_path}")
            return [], [], []
        try:
            df = pd.read_csv(self.csv_path)
            questions = list(df["User Input"])
            references = list(df["Reference"])
            answers = list(df["Response"])
            logging.info(f"CSV data loaded from {self.csv_path}")
            return questions, references, answers
        except Exception as e:
            logging.error(f"Error loading CSV data: {str(e)}")
            return [], [], []


class Evaluator:
    """
    Evaluates question-answer pairs using the retriever, an LLM, and evaluation metrics.
    """
    def __init__(self, retriever, output_csv="evaluation_results.csv", output_json="evaluation_results.json"):
        self.retriever = retriever
        self.output_csv = output_csv
        self.output_json = output_json
        self._initialize_output_files()

        # Initialize LLM and embedding wrappers for evaluation
        self.eval_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.evaluator_llm = LangchainLLMWrapper(self.eval_llm)
        self.embeddings = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
        )
        logging.info("Evaluator initialized.")

    def _initialize_output_files(self):
        """Creates CSV output file with header if it does not exist."""
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "User Input", "Retrieved Contexts", "Response", "Reference", 
                    "Context Recall", "Faithfulness", "Semantic Similarity", "Answer Correctness"
                ])
            logging.info(f"CSV output file created: {self.output_csv}")

    def evaluate_all(self, questions, references, answers):
        print("Starting evaluation...")
        for idx, (query, ref, rag_answer) in enumerate(zip(questions, references, answers)):
            try:
                print(f"Processing {idx+1}/{len(questions)}: {query[:50]}...")
                self.evaluate_single(query, ref, rag_answer)
                print(f"Evaluation saved for query {idx+1}")

                # Sleep after every 3 evaluations to avoid rate limits
                if (idx + 1) % 3 == 0:
                    print("Sleeping for 60 seconds to avoid rate limits...")
                    time.sleep(60)
            except Exception as e:
                logging.error(f"Error processing query {idx+1}: {str(e)}")
        logging.info("Evaluation completed for all queries.")

    def evaluate_single(self, query, reference, rag_answer):
        """Evaluates a single question-answer pair and saves the results."""
        # Retrieve relevant documents for the query
        relevant_docs = self.retriever.invoke(query)
        dataset = [{
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": rag_answer,
            "reference": reference,
        }]

        evaluation_dataset = EvaluationDataset.from_list(dataset)
        
        # Evaluate using the RAGAS evaluation function
        result = evaluate(
            dataset=evaluation_dataset,
            embeddings=self.embeddings,
            metrics=[
                LLMContextRecall(max_retries=3), 
                Faithfulness(max_retries=3),
                SemanticSimilarity(), 
                AnswerCorrectness(max_retries=3)
            ],
            llm=self.evaluator_llm,
        )

        eval_scores = {
            "context_recall": result.scores[0].get("context_recall", 0),
            "faithfulness": result.scores[0].get("faithfulness", 0),
            "semantic_similarity": result.scores[0].get("semantic_similarity", 0),
            "answer_correctness": result.scores[0].get("answer_correctness", 0)
        }

        # Append evaluation results to dataset and save
        dataset[0]["evaluation_result"] = eval_scores
        self.save_result_incrementally(dataset[0])

    def save_result_incrementally(self, data_item):
        """Appends a single evaluation result to CSV and JSON output files."""
        try:
            # Append to CSV
            with open(self.output_csv, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    data_item["user_input"],
                    " | ".join(data_item["retrieved_contexts"]),
                    data_item["response"],
                    data_item["reference"],
                    data_item["evaluation_result"]["context_recall"],
                    data_item["evaluation_result"]["faithfulness"],
                    data_item["evaluation_result"]["semantic_similarity"],
                    data_item["evaluation_result"]["answer_correctness"]
                ])
            logging.info(f"Saved result to CSV: {self.output_csv}")

            # Load existing JSON data or create new list
            if not os.path.exists(self.output_json):
                json_data = []
            else:
                with open(self.output_json, "r", encoding="utf-8") as file:
                    json_data = json.load(file)
            
            json_data.append(data_item)
            with open(self.output_json, "w", encoding="utf-8") as file:
                json.dump(json_data, file, indent=4, ensure_ascii=False)
            logging.info(f"Saved result to JSON: {self.output_json}")

        except Exception as e:
            logging.error(f"Error saving result: {str(e)}")



# Main execution workflow
def main():
    # Pinecone setup
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = "rag-final"
    pinecone_handler = PineconeHandler(api_key=pinecone_api_key, index_name=index_name)

    # Data loading from CSV
    csv_path = "RAG_Project/Golden_Dataset_19_20.csv"
    data_loader = DataLoader(csv_path=csv_path)
    questions, references, answers = data_loader.load_csv_data()

    # Initialize evaluator with the retriever from PineconeHandler
    evaluator = Evaluator(retriever=pinecone_handler.retriever,
                          output_csv="RAGAS_Result.csv",
                          output_json="RAGAS_Result.json")
    
    # Run evaluation
    evaluator.evaluate_all(questions, references, answers)


if __name__ == "__main__":
    main()
