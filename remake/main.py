import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from extract_doc import extract_text

# Définir le chemin d'accès aux modèles pré-entraînés
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Charger le tokenizer et le modèle pour BERT
tokenizer_bert = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "bert-base-uncased"))
model_bert = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "bert-base-uncased"))

# Charger le tokenizer et le modèle pour RoBERTa
tokenizer_roberta = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "roberta-base"))
model_roberta = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "roberta-base"))

# Charger le tokenizer et le modèle pour DistilBERT
tokenizer_distilbert = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "distilbert-base-uncased"))
model_distilbert = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, "distilbert-base-uncased"))

# Définir la fonction pour classer le texte en utilisant un modèle donné
def classify_text(model, tokenizer, text):
    # Tokeniser le texte en entrée
    inputs = tokenizer(text, return_tensors="pt")

    # Évaluer le modèle sur les entrées tokenisées
    outputs = model(**inputs)

    # Récupérer les probabilités prédites pour chaque classe
    probs = torch.softmax(outputs.logits, dim=1)

    # Renvoie la classe prédite avec la probabilité la plus élevée
    return probs.argmax().item()

# Demander à l'utilisateur de choisir un fichier à analyser
file_path = input("Entrez le chemin d'accès du fichier que vous voulez analyser : ")

# Extraire le texte du fichier
text = extract_text(file_path)

# Afficher les options de modèle à l'utilisateur
print("\nChoisissez un modèle pour classer le texte : ")
print("1. BERT")
print("2. RoBERTa")
print("3. DistilBERT")

# Demander à l'utilisateur de choisir un modèle
model_choice = int(input("Entrez le numéro du modèle que vous voulez utiliser : "))

# Sélectionner le modèle en fonction du choix de l'utilisateur
if model_choice == 1:
    model = model_bert
    tokenizer = tokenizer_bert
elif model_choice == 2:
    model = model_roberta
    tokenizer = tokenizer_roberta
elif model_choice == 3:
    model = model_distilbert
    tokenizer = tokenizer_distilbert
else:
    print("Choix de modèle invalide. Utilisez l'un des numéros de modèle répertoriés.")
    exit()

# Classer le texte en utilisant le modèle sélectionné
label = classify_text(model, tokenizer, text)

# Afficher le résultat de la classification
print(f"Le texte a été classé dans la catégorie {label}.")
