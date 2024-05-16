import os
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def read_pdf_file(file_path):
    text = ""
    with fitz.open(file_path) as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text.split('\n')

def read_documents(file_or_directory):
    documents = []
    if os.path.isfile(file_or_directory):
        if file_or_directory.endswith('.txt'):
            documents.extend(read_text_file(file_or_directory))
        elif file_or_directory.endswith('.pdf'):
            documents.extend(read_pdf_file(file_or_directory))
    elif os.path.isdir(file_or_directory):
        for file_name in os.listdir(file_or_directory):
            file_path = os.path.join(file_or_directory, file_name)
            if file_name.endswith('.txt'):
                documents.extend(read_text_file(file_path))
            elif file_name.endswith('.pdf'):
                documents.extend(read_pdf_file(file_path))
    return documents

# Charger un modèle de Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Spécifier le chemin du fichier ou du répertoire contenant les documents
file_or_directory = 'document.pdf'

# Lire les documents à partir du fichier ou du répertoire
documents = read_documents(file_or_directory)

# Encoder les documents
embeddings = model.encode(documents, convert_to_numpy=True)

# Sauvegarder les documents avec embeddings dans un fichier 'doc_scanned.txt'
with open('doc_scanned.txt', 'w', encoding='utf-8') as f:
    for doc in documents:
        f.write(f"{doc.strip()}\n")

print('Votre document a été scanné avec succès')
