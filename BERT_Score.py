import pandas as pd
from bert_score import score

# Load your combined CSV file (adjust the filename if necessary)
df = pd.read_csv("combined.csv")

# Extract the candidate answers (from your RAG-LLM) and the reference answers (your responses)
candidates = df['rag_answer'].astype(str).tolist()
references = df['answer'].astype(str).tolist()

# Compute BERTScore; this returns precision, recall, and F1 scores for each answer pair.
P, R, F1 = score(candidates, references, lang="en", verbose=True)

# Add the F1 score as a new column in the DataFrame.
df['bert_f1'] = F1.tolist()

# Save the evaluation results to a new CSV file.
df.to_csv("BERT_results.csv", index=False)

print("BERTScore evaluation completed. Results saved to 'BERT_results.csv'.")
