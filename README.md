# RAG_Project

## Project Overview
This project involves building a **Retrieval-Augmented Generation (RAG) pipeline** using **Gemini** LLM. The workflow includes **data scraping, chunking, embedding generation, indexing, query retrieval, response generation, and evaluation** using **BLEU and BERT scores**. A **Streamlit** frontend is developed for user interaction, and all processes are logged for debugging and analysis.

## Project Workflow
### 1. Data Scraping
- Collected **American movies released between 2019-2023**.
- Scraped all relevant movie data present in text format. 
- Stored the data in folders **[movies_2019](movies_2019), [movies_2020](movies_2020), [movies_2021](movies_2021), [movies_2022](movies_2022), [movies_2023](movies_2023)**.

### 2. Chunking
- Processed the movie dataset by chunking textual information to ensure optimal embedding.
- Applied **text chunking techniques**, breaking large texts into meaningful chunks.
- Stored chunked data in **[chunked_data.pkl](chunked_data.pkl)**.

### 3. Embedding & Indexing
- **Embedding Generation:**
  - Generated embeddings for chunked text using a **pre-trained embedding model**.
  - Used **(FILL IN EMBEDDING MODEL NAME: Sentence Transformers: all-mpnet-base-v2)**.
  - Code for this is available in file: **[Embedding_generation_and_indexing.py](Embedding_generation_and_indexing.py)**

- **Indexing with Pinecone:**
  - Created an **index in Pinecone** for efficient similarity search and retrieval.
  - Indexed **114768** items.
  - Stored indexed data in **rag**.

### 4. Retrieval-Augmented Generation (RAG) Pipeline
- Used **Gemini** as the **LLM** for **retrieving relevant chunks and generating responses**.
- Retrieval Mechanism:
  - User query → **Vector search in Pinecone** → **Relevant document retrieval** → **Final response generation using Gemini**.
- Implemented the pipeline in **[RAG_Implementation.ipynb](RAG_Implementation.ipynb)**.

### 5. Frontend Development
- Developed an interactive **Streamlit application**.
- Allows users to **query movie-related information** and get AI-generated responses.
- Integrated UI features for **user input, response display, and performance analysis**.
- Frontend code is in **[app.py](app.py)**.

### 6. Logging
- Maintained a **.log file** to track execution, errors, and query performance.
- Stores details like **timestamps, query processing time, retrieval steps, and errors**.
- Log file is stored as **[qa_interactions.log](qa_interactions.log)**.

### 7. Test Case Generation
- Developed a **Python script** to generate test cases for evaluating the RAG pipeline.
- Stored test cases in **(test_cases.csv)**.
- Test case generation script is **[generate_test_cases.py](generate_test_cases.py)**.

### 8. Comparison with GPT
- The generated test cases were run through **both the RAG pipeline and a GPT-based model**.
- Compared **response accuracy, coherence, and retrieval effectiveness**.
- Evaluation results were stored in **[GPT_Answers.csv](GPT_Answers.csv)**.

### 9. Evaluation Metrics
- Used **BLEU Score** to measure text similarity between RAG-generated responses and reference answers.
- Used **BERT Score** to evaluate contextual similarity for better accuracy.
- Evaluation was performed using **[BERT_Score.py](BERT_Score.py)** and **[BLEU_response.py](BLEU_response.py)**.

## Technologies Used
- **Python**: Primary programming language.
- **Pinecone**: Vector database for indexing and retrieval.
- **Gemini**: LLM for RAG pipeline.
- **Streamlit**: Frontend development.
- **BLEU & BERT Score**: Evaluation metrics.
- **Logging Module**: Used to track processes.

## How to Run the Project
### Prerequisites
- Install required libraries:
  ```bash
  pip install pinecone-client google-generativeai streamlit numpy scikit-learn transformers nltk
  ```

### Steps to Execute
1. **Run Data Processing (Chunking):**
   ```bash
   python (Chunking.py)
   ```
2. **Generate Embeddings & Indexing in Pinecone:**
   ```bash
   python (Embedding_generation_and_indexing.py)
   ```
3. **Run RAG Pipeline:**
   ```bash
   python (RAG_Implementation.ipynb)
   ```
4. **Launch Frontend:**
   ```bash
   streamlit run (app.py)
   ```

## Output & Results
- **Generated responses from the RAG pipeline.**
- **Comparisons with GPT outputs.**
- **Final evaluation scores (BLEU & BERT) for performance analysis.**

## Log File
- A `.log` file **[qa_interactions.log](qa_interactions.log)** tracks system execution, errors, and response time details.



