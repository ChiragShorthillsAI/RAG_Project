import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

# Load your CSV file containing: question, rag_answer, answer (your response)
df = pd.read_csv("combined.csv")

def evaluate_bleu(reference, hypothesis):
    # Simple tokenization (you might want to use a more robust tokenizer)
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    # Calculate BLEU score with reference as a list of one reference
    score = sentence_bleu([ref_tokens], hyp_tokens)
    return score

# Compute BLEU score for each row comparing your answer (reference) to the rag_answer (hypothesis)
df['bleu_score'] = df.apply(lambda row: evaluate_bleu(row['answer'], row['rag_answer']), axis=1)

# Save the evaluation results
df.to_csv("BLEU_result.csv", index=False)
print(df[['question', 'bleu_score']].head())
