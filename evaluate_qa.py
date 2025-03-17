import csv
import os
import time
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

# Import your local RAG chain from app.py
from RAG_Project.RAG_Implementation import rag_system

# Wrap the local RAG chain invocation with exponential backoff to handle 429 errors.
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(ResourceExhausted)
)

# def safe_rag_invoke(question: str) -> dict:
#     return rag_system.ask_question({"input": question})

def safe_rag_invoke(question: str) -> dict:
    response = rag_system.ask_question(question)  # Ensure this returns a valid dictionary
    print(f"DEBUG - Response for '{question}':", response)  # Debugging line
    return response


def get_rag_answer(question: str) -> str:
    """Get answer from the local RAG chain with newlines removed."""
    try:
        response = safe_rag_invoke(question)
        answer = response.get("answer", "No answer available.")
        return answer.replace("\n", " ").replace("\r", " ")
    except Exception as e:
        return f"Error retrieving answer: {str(e)}"

def main():
    load_dotenv()
    input_file = "test_cases.csv"       # CSV file with a "question" column
    output_file = "evaluation_results.csv"

    seen_questions = set()

    with open(input_file, newline='', encoding='utf-8') as csv_in, \
         open(output_file, "w", newline='', encoding='utf-8') as csv_out:
        reader = csv.DictReader(csv_in)
        fieldnames = ["question", "rag_answer"]
        writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            question = row.get("question", "").strip()
            if not question or question in seen_questions:
                continue
            seen_questions.add(question)

            rag_answer = get_rag_answer(question)

            writer.writerow({
                "question": question,
                "rag_answer": rag_answer
            })

            print(f"Processed: {question}")
            # Increase sleep duration to reduce request rate (adjust as needed)
            time.sleep(5)

if __name__ == "__main__":
    main()
