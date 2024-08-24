import sys
import json
import torch
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder

# Sample dataset
data = [
    {"symptom": "fever cough sore throat", "disease": "Flu", "treatment": "Rest, Fluids", "diet": "Chicken soup"},
    {"symptom": "fever rash joint pain", "disease": "Dengue", "treatment": "Fluid intake, Pain relievers", "diet": "High fluid diet"},
    {"symptom": "headache dizziness nausea", "disease": "Migraine", "treatment": "Pain relievers, Rest", "diet": "Hydration"},
    # Additional entries here...
]

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Extract the embeddings for the CLS token, which represents the sentence embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.numpy().flatten()

# Prepare dataset
symptoms = [item["symptom"] for item in data]
diseases = [item["disease"] for item in data]

# Convert symptoms to BERT embeddings
X = [get_sentence_embedding(symptom) for symptom in symptoms]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(diseases)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Get input symptom from the command-line argument
user_symptom = sys.argv[1] if len(sys.argv) > 1 else ""

try:
    if not user_symptom:
        raise ValueError("No symptom provided")

    # Transform user symptom into BERT embedding
    user_embedding = get_sentence_embedding(user_symptom)

    # Predict the disease
    predicted_label = model.predict([user_embedding])[0]
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]

    # Find matching treatment and diet
    for entry in data:
        if entry["disease"] == predicted_disease:
            result = {
                "disease": entry["disease"],
                "treatment": entry["treatment"],
                "diet": entry["diet"]
            }
            break
    else:
        result = {"disease": "Unknown", "treatment": "Consult a doctor", "diet": "Balanced diet"}

except Exception as e:
    result = {"error": str(e)}

# Output the result as JSON
print(json.dumps(result))
