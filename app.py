from transformers import pipeline
import torch

# Charger le modèle de Hugging Face pour la génération de texte
generator = pipeline('text-generation', model='gpt2')

def read_document(file_path):
    with open(file_path, 'rb') as f:
        content = f.read().decode('latin-1')  # Utilisez 'latin-1' si 'utf-8' cause des erreurs
    return content

def generate_answer(prompt, question, document_content):
    full_prompt = f"{document_content}\n\n{prompt}\n\nQuestion: {question}\nAnswer:"
    response = generator(full_prompt, max_length=200, num_return_sequences=1)
    return response[0]['generated_text'].strip()

if __name__ == "__main__":
    # Demander le chemin du document
    document_path = input("Entrez le chemin du document: ")
    document_content = read_document(document_path)

    # Demander le prompt et la question
    prompt = input("Entrez votre prompt: ")
    question = input("Posez votre question: ")

    # Générer la réponse
    generated_answer = generate_answer(prompt, question, document_content)

    # Afficher la réponse dans le terminal
    print("\nQuestion:", question)
    print("Réponse:", generated_answer)
