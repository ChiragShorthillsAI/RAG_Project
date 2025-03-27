
# RAG & Evaluation Project

This project implements a Retrieval-Augmented Generation (RAG) system with end-to-end data scraping, indexing, embedding, answer generation, and multi-parameter evaluation. It also provides a Streamlit-based interface to interact with the RAG system.

## Table of Contents

- [Overview](#overview)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage Instructions](#usage-instructions)
- [Evaluation Metrics](#evaluation-metrics)
- [Dependencies](#dependencies)
- [Configuration](#configuration)


## Overview

This project covers the complete workflow of scraping data, indexing and embedding text for a RAG system, generating answers using a large language model, and evaluating the answers on multiple parameters.

## Workflow

1. **Scraping Data:**  
   Run `Scraping.py` to scrape movie data from Wikipedia. Two folders, `Movies_2019` and `Movies_2020`, are generated. Each folder contains the scraped movie data saved in `.txt` format.

2. **Indexing & Embedding:**  
   Execute `RAG_Implementation.py` to load the scraped text, split the documents, generate embeddings, and create an index (using Pinecone) for the RAG system.

3. **Interactive Q&A Interface:**  
   Launch `app.py`, a Streamlit-based frontend that uses the indexed data to provide an interactive question-answering interface.

4. **RAG Answer Generation & Evaluation:**  
   Run `RAGAS.py` to generate answers using the `llama-3.1-8b-instant` model. The generated answers are evaluated by the `gemini-2.0-flash` model based on four parameters: **Context Recall**, **Faithfulness**, **Semantic Similarity**, and **Answer Correctness**. The metrics and the generated RAG responses are saved in both `RAGAS_Result.json` and `RAGAS_Result.csv`.

5. **NLI Evaluation:**  
   Run `NLI_score.py` to further evaluate the quality of the generated answers using the `roberta-large-mnli` model.

## Project Structure

```
├── Scraping.py                # Scrapes movie data from Wikipedia and saves it as text files
├── Movies_2019/               # Folder containing scraped movie data for 2019 (generated by Scraping.py)
├── Movies_2020/               # Folder containing scraped movie data for 2020 (generated by Scraping.py)
├── RAG_Implementation.py      # Implements indexing, embedding, and sets up the RAG system
├── app.py                     # Streamlit-based interface for interacting with the RAG system
├── RAGAS.py                   # Generates RAG answers using llama-3.1-8b-instant and evaluates them with gemini-2.0-flash
├── RAGAS_Result.json          # JSON file containing evaluation metrics and RAG responses
├── RAGAS_Result.csv           # CSV file containing evaluation metrics and RAG responses
└── NLI_score.py               # Evaluates generated answers using the roberta-large-mnli model for NLI scoring
```

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**

   Ensure you have a `requirements.txt` file and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

### 1. Scraping Movie Data

- **Run the Scraping Script:**

  ```bash
  python Scraping.py
  ```

  This will create two folders (`Movies_2019` and `Movies_2020`) with text files containing the scraped movie data.

### 2. Indexing & Embedding

- **Set Up the RAG System:**

  ```bash
  python RAG_Implementation.py
  ```

  This script will load the text files, generate embeddings, and create a vector index for the RAG system.

### 3. Interactive Q&A Interface

- **Launch the Streamlit App:**

  ```bash
  streamlit run app.py
  ```

  Access the Q&A interface via your web browser to interact with the RAG system.

### 4. Answer Generation & Evaluation

- **Generate and Evaluate RAG Answers:**

  ```bash
  python RAGAS.py
  ```

  The script generates answers using `llama-3.1-8b-instant`, evaluates them with `gemini-2.0-flash`, and saves the evaluation metrics and responses in `RAGAS_Result.json` and `RAGAS_Result.csv`.

### 5. NLI Evaluation

- **Run NLI Evaluation:**

  ```bash
  python NLI_score.py
  ```

  This will evaluate the generated answers using the `roberta-large-mnli` model.

## Evaluation Metrics

- **Context Recall:** Measures how well the generated answer covers the relevant context.
- **Faithfulness:** Assesses whether the answer is consistent with the provided context.
- **Semantic Similarity:** Evaluates the similarity between the generated answer and the ground truth.
- **Answer Correctness:** Checks the accuracy of the generated answer.

## Dependencies

Key libraries used include:

- **Streamlit** – For building the web-based Q&A interface.
- **Pinecone Client & LangChain** – For vector storage, retrieval, and document handling.
- **Google Generative AI Libraries** – For embedding and answer generation.
- **llama-3.1-8b-instant** - For Response Generaion
- **Gemini-2.0-flash** - For Evaluation of RAG
- **Transformers (Hugging Face)** – For NLI evaluation using the RoBERTa model.
- **Pandas & NumPy** – For data manipulation.
- **Requests & BeautifulSoup** – For web scraping.


## Configuration

Create a `.env` file in the project root with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key  # Only required if using ChatGroq in RAG_Implementation.py
```



## Log File
- A `.log` file **[qa_interactions.log](qa_interactions.log)** tracks system execution, errors, and response time details.



