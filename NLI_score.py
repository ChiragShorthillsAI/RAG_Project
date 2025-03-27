import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pandas as pd

# Load the dataset
file_path = "final_nli_set.csv"
df = pd.read_csv(file_path)

# convert to list of dicts if needed
golden_data = df.to_dict(orient="records")

if golden_data is None:
    raise ValueError("No variable named `golden_data` found in combine_golden.py")

# Load NLI model and tokenizer
model_name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Inference helper
def classify_nli(premise, hypothesis):
    inputs = tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze()
    labels = ['contradiction', 'neutral', 'entailment']
    pred_label = labels[torch.argmax(probs).item()]
    return pred_label, probs.tolist()

# Evaluate each (ground truth, rag answer) pair
results = []
for item in golden_data:
    gt = item.get("answer", "")
    rag = item.get("response", "")
    label, prob = classify_nli(gt, rag)
    results.append({
        "query": item.get("query", ""),
        "answer": gt,
        "response": rag,
        "nli_result": label,
        "confidence": prob
    })

# Display results
import pandas as pd
df = pd.DataFrame(results)
# Save results to CSV
df.to_csv("nli_evaluation_result_final.csv", index=False)

# Printing the first few rows
print(df.head())
