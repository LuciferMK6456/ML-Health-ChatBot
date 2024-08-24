# backend/ml_model.py

import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset of symptoms and their corresponding disease, treatment, and diet
TREATMENT_REST_FLUIDS = "Rest, Fluids"
TREATMENT_PAIN_RELAISERS_REST = "Pain relievers, Rest"
TREATMENT_ANTIBIOTICS = "Antibiotics"
TREATMENT_ANTIMALARIAL_DRUGS = "Antimalarial Drugs"
TREATMENT_BALANCED_DIET="Balanced Diet"
DIET_FLUIDS_SOFT = "Fluids, Soft diet"

data = [
    {"symptom": "fever cough sore throat", "disease": "Flu", "treatment": TREATMENT_REST_FLUIDS, "diet": "Chicken soup"},
    {"symptom": "fever rash joint pain", "disease": "Dengue", "treatment": "Fluid intake, Pain relievers", "diet": "High fluid diet"},
    {"symptom": "headache dizziness nausea", "disease": "Migraine", "treatment": TREATMENT_PAIN_RELAISERS_REST, "diet": "Hydration"},
    {"symptom": "fever chills night sweats", "disease": "Malaria", "treatment": "Antimalarial drugs", "diet": "Light diet, fluids"},
    {"symptom": "chest pain shortness of breath", "disease": "Angina", "treatment": "Nitrates, Rest", "diet": "Low-sodium diet"},
    {"symptom": "fever fatigue cough", "disease": "Pneumonia", "treatment": "Antibiotics, Rest", "diet": "Light diet, warm fluids"},
    {"symptom": "weight loss frequent urination thirst", "disease": "Diabetes", "treatment": "Insulin, Diet control", "diet": "Low-sugar diet"},
    {"symptom": "itching sneezing runny nose", "disease": "Allergic Rhinitis", "treatment": "Antihistamines", "diet": "Avoid allergens"},
    {"symptom": "fever body aches dry cough", "disease": "COVID-19", "treatment": "Self-isolation, Rest", "diet": "Hydration, balanced diet"},
    {"symptom": "abdominal pain diarrhea fever", "disease": "Gastroenteritis", "treatment": "Rehydration, Antidiarrheal", "diet": "BRAT diet (bananas, rice, applesauce, toast)"},
    {"symptom": "fever cough fatigue", "disease": "Common Cold", "treatment": TREATMENT_REST_FLUIDS, "diet": "Warm fluids"},
    {"symptom": "cough difficulty breathing chest tightness", "disease": "Asthma", "treatment": "Inhaler, Bronchodilators", "diet": "Anti-inflammatory foods"},
    {"symptom": "fatigue pale skin shortness of breath", "disease": "Anemia", "treatment": "Iron supplements", "diet": "Iron-rich foods"},
    {"symptom": "frequent urination burning sensation", "disease": "Urinary Tract Infection", "treatment": "Antibiotics", "diet": "Cranberry juice, Water"},
    {"symptom": "nausea vomiting right upper abdominal pain", "disease": "Gallstones", "treatment": "Surgery, Pain relief", "diet": "Low-fat diet"},
    {"symptom": "blurred vision eye pain headache", "disease": "Glaucoma", "treatment": "Eye drops, Surgery", "diet": "Leafy greens, Omega-3"},
    {"symptom": "joint pain stiffness swelling", "disease": "Arthritis", "treatment": "Anti-inflammatory drugs", "diet": "Omega-3, Anti-inflammatory foods"},
    {"symptom": "fever abdominal cramps vomiting", "disease": "Food Poisoning", "treatment": "Rehydration, Antiemetics", "diet": "BRAT diet"},
    {"symptom": "fatigue muscle pain difficulty breathing", "disease": "Chronic Fatigue Syndrome", "treatment": "Therapy, Pain relief", "diet": "Balanced diet, Hydration"},
    {"symptom": "fever muscle pain headache", "disease": "Typhoid", "treatment": "Antibiotics", "diet": "Fluids, Soft foods"},
    {"symptom": "fever sore throat body aches", "disease": "Strep Throat", "treatment": "Antibiotics", "diet": "Soft foods, Fluids"},
    {"symptom": "itching redness swelling", "disease": "Eczema", "treatment": "Moisturizers, Steroid creams", "diet": "Anti-inflammatory foods"},
    {"symptom": "frequent thirst dry mouth headache", "disease": "Dehydration", "treatment": "Rehydration", "diet": "Electrolyte-rich drinks"},
    {"symptom": "sharp abdominal pain bloating gas", "disease": "Irritable Bowel Syndrome", "treatment": "Diet modification", "diet": "Low-FODMAP diet"},
    {"symptom": "headache neck pain sensitivity to light", "disease": "Meningitis", "treatment": "Antibiotics, Hospitalization", "diet": DIET_FLUIDS_SOFT},
    {"symptom": "shortness of breath chest pain cough", "disease": "Pulmonary Embolism", "treatment": "Anticoagulants", "diet": "Light diet"},
    {"symptom": "fever severe headache stiff neck", "disease": "Encephalitis", "treatment": "Antiviral drugs", "diet": DIET_FLUIDS_SOFT},
    {"symptom": "fever fatigue joint pain", "disease": "Chikungunya", "treatment": TREATMENT_PAIN_RELAISERS_REST, "diet": "Hydration, Balanced diet"},
    {"symptom": "weakness fatigue chest pain", "disease": "Heart Disease", "treatment": "Lifestyle changes, Medication", "diet": "Heart-healthy diet"},
    {"symptom": "headache fever nausea vomiting", "disease": "Brain Tumor", "treatment": "Surgery, Radiation", "diet": TREATMENT_BALANCED_DIET},
    {"symptom": "fever chills vomiting", "disease": "Sepsis", "treatment": "Antibiotics, Fluids", "diet": "Soft diet"},
    {"symptom": "swelling redness warmth", "disease": "Cellulitis", "treatment": "Antibiotics", "diet": TREATMENT_BALANCED_DIET},
    {"symptom": "fever cough fatigue", "disease": "Bronchitis", "treatment": TREATMENT_REST_FLUIDS, "diet": "Warm fluids"},
    {"symptom": "headache ringing in ears dizziness", "disease": "Tinnitus", "treatment": "Sound therapy, Cognitive therapy", "diet": TREATMENT_BALANCED_DIET},
    {"symptom": "rash joint pain headache", "disease": "Zika Virus", "treatment": "Rest, Pain relief", "diet": "Hydration, Balanced diet"},
    {"symptom": "fever cough shortness of breath", "disease": "Tuberculosis", "treatment": "Antibiotics", "diet": "High-protein diet"},
    {"symptom": "frequent urination thirst weight loss", "disease": "Type 2 Diabetes", "treatment": "Diet control, Medication", "diet": "Low-carb diet"},
    {"symptom": "blurred vision thirst fatigue", "disease": "Hyperglycemia", "treatment": "Insulin therapy", "diet": "Low-sugar diet"},
    {"symptom": "fever chills headache", "disease": "Leptospirosis", "treatment": "Antibiotics", "diet": DIET_FLUIDS_SOFT},
    {"symptom": "weakness dizziness fatigue", "disease": "Low Blood Pressure", "treatment": "Increase salt intake", "diet": "Salt-rich foods"},
    {"symptom": "fever abdominal pain diarrhea", "disease": "Salmonella Infection", "treatment": "Antibiotics", "diet": "BRAT diet"},
    {"symptom": "sore throat fatigue loss of appetite", "disease": "Mononucleosis", "treatment": TREATMENT_REST_FLUIDS, "diet": "Soft foods"},
    {"symptom": "fever weight loss night sweats", "disease": "Lymphoma", "treatment": "Chemotherapy", "diet": TREATMENT_BALANCED_DIET},
    {"symptom": "cough shortness of breath fatigue", "disease": "Emphysema", "treatment": "Bronchodilators, Oxygen therapy", "diet": "Anti-inflammatory diet"},
    {"symptom": "fever weight loss cough", "disease": "Lung Cancer", "treatment": "Chemotherapy, Surgery", "diet": "High-protein diet"},
    {"symptom": "fever headache fatigue", "disease": "Hepatitis B", "treatment": "Antiviral drugs", "diet": "Low-fat diet"},
    {"symptom": "sore throat cough fever", "disease": "Tonsillitis", "treatment": "Antibiotics", "diet": "Soft foods, Fluids"},
    {"symptom": "stomach pain bloating diarrhea", "disease": "Crohn's Disease", "treatment": "Anti-inflammatory drugs", "diet": "Low-fiber diet"},
    {"symptom": "dizziness fainting fatigue", "disease": "Anxiety", "treatment": "Therapy, Medication", "diet": TREATMENT_BALANCED_DIET},
    {"symptom": "muscle pain joint stiffness fatigue", "disease": "Fibromyalgia", "treatment": "Pain relievers, Therapy", "diet": TREATMENT_BALANCED_DIET},
    {"symptom": "chest pain rapid heartbeat shortness of breath", "disease": "Panic Attack", "treatment": "Relaxation techniques", "diet": TREATMENT_BALANCED_DIET},
]

# Extract symptoms and diseases
symptoms = [item["symptom"] for item in data]
diseases = [item["disease"] for item in data]

# Train TF-IDF and Naive Bayes model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(symptoms)
model = MultinomialNB()
model.fit(X, diseases)

# Get input from command line argument
user_symptom = sys.argv[1] if len(sys.argv) > 1 else ""

if not user_symptom:
    # Handle case where no input is provided
    result = {"disease": "Unknown", "treatment": "Consult a doctor", "diet": "Balanced diet"}
else:
    # Make prediction
    user_input_vector = vectorizer.transform([user_symptom])
    predicted_disease = model.predict(user_input_vector)[0]

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
        # If no match found in the dataset
        result = {"disease": "Unknown", "treatment": "Consult a doctor", "diet": "Balanced diet"}

# Output JSON
print(json.dumps(result))