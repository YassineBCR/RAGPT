from flask import Flask, request, render_template
from extract_doc import extract_text_from_pdf, extract_text_from_docx, extract_text_from_xlsx, extract_text_from_pptx, extract_text_from_txt
from model import tokenizer_bert, model_bert, tokenizer_roberta, model_roberta, tokenizer_distilbert, model_distilbert, tokenizer_xlnet, model_xlnet, tokenizer_electra, model_electra, tokenizer_gpt, model_gpt, tokenizer_gpt2, model_gpt2, tokenizer_mistral, model_mistral, tokenizer_llama, model_llama
import os
import torch

app = Flask(__name__)

# Route pour afficher le formulaire de téléchargement de fichier
@app.route('/')
def index():
    return render_template('index.html')

# Route pour traiter le fichier téléchargé et répondre à la question de l'utilisateur
@app.route('/ask', methods=['POST'])
def ask():
    # Vérifier si l'utilisateur a téléchargé un fichier
    if 'file' not in request.files:
        return render_template('error.html', message='Veuillez télécharger un fichier.')
    file = request.files['file']
    # Vérifier si le fichier est vide
    if file.filename == '':
        return render_template('error.html', message='Veuillez télécharger un fichier.')
    # Déterminer le type de fichier en fonction de l'extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    # Extraire le texte brut à partir du fichier en fonction du type de fichier
    if file_ext == ".pdf":
        extracted_text = extract_text_from_pdf(file)
    elif file_ext == ".docx":
        extracted_text = extract_text_from_docx(file)
    elif file_ext == ".xlsx" or file_ext == ".xls":
        extracted_text = extract_text_from_xlsx(file)
    elif file_ext == ".pptx" or file_ext == ".ppt":
        extracted_text = extract_text_from_pptx(file)
    elif file_ext == ".txt":
        extracted_text = extract_text_from_txt(file)
    else:
        return render_template('error.html', message='Le type de fichier n\'est pas pris en charge.')
    # Récupérer la question de l'utilisateur à partir de la requête
    question = request.form.get('question')
    # Récupérer le nom du modèle sélectionné par l'utilisateur à partir de la requête
    model_name = request.form.get('model')
    # Sélectionner le modèle et le tokenizer correspondants en fonction du nom du modèle
    if model_name == 'bert':
        tokenizer = tokenizer_bert
        model = model_bert
    elif model_name == 'roberta':
        tokenizer = tokenizer_roberta
        model = model_roberta
    elif model_name == 'distilbert':
        tokenizer = tokenizer_distilbert
        model = model_distilbert
    elif model_name == 'xlnet':
        tokenizer = tokenizer_xlnet
        model = model_xlnet
    elif model_name == 'electra':
        tokenizer = tokenizer_electra
        model = model_electra
    elif model_name == 'gpt':
        tokenizer = tokenizer_gpt
        model = model_gpt
    elif model_name == 'gpt2':
        tokenizer = tokenizer_gpt2
        model = model_gpt2
    elif model_name == 'mistral':
        tokenizer = tokenizer_mistral
        model = model_mistral
    elif model_name == 'llama':
        tokenizer = tokenizer_llama
        model = model_llama
    else:
        return render_template('error.html', message='Le modèle sélectionné n\'est pas disponible.')
    # Utiliser le modèle pour répondre à la question en fonction du type de modèle
    if model_name in ['bert', 'roberta', 'distilbert', 'xlnet', 'electra']:
        # Tokenizer la question et le texte brut
        question_tokens = tokenizer.tokenize(question)
        text_tokens = tokenizer.tokenize(extracted_text)

        # Ajouter les tokens spéciaux [CLS] et [SEP] au début et à la fin de la question et du texte brut
        question_tokens = ['[CLS]'] + question_tokens + ['[SEP]']
        text_tokens = ['[CLS]'] + text_tokens + ['[SEP]']

        # Tronquer ou padder la question et le texte brut pour qu'ils aient la même longueur
        question_length = len(question_tokens)
        text_length = len(text_tokens)
        max_length = max(question_length, text_length)
        if question_length < max_length:
            question_tokens += ['[PAD]'] * (max_length - question_length)
        if text_length < max_length:
            text_tokens += ['[PAD]'] * (max_length - text_length)

        # Convertir les tokens en indices et créer les tensors d'entrée pour le modèle
        question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
        text_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        input_ids = torch.tensor([question_ids + text_ids]).unsqueeze(0)
        attention_mask = torch.tensor([[1] * (question_length + text_length) + [0] * (max_length - (question_length + text_length))]).unsqueeze(0)

        # Utiliser le modèle pour prédire la réponse à la question
        outputs = model(input_ids, attention_mask=attention_mask)
        start_indices = torch.argmax(outputs.start_logits, dim=-1)
        end_indices = torch.argmax(outputs.end_logits, dim=-1)
        predicted_answer = extracted_text[start_indices.item():end_indices.item()]
    else:
        # Tokenizer la question et le texte brut
        question_tokens = tokenizer.encode(question, add_special_tokens=True)
        text_tokens = tokenizer.encode(extracted_text, add_special_tokens=True)

        # Tronquer ou padder la question et le texte brut pour qu'ils aient la même longueur
        question_length = len(question_tokens)
        text_length = len(text_tokens)
        max_length = max(question_length, text_length)
        if question_length < max_length:
            question_tokens += [tokenizer.pad_token_id] * (max_length - question_length)
        if text_length < max_length:
            text_tokens += [tokenizer.pad_token_id] * (max_length - text_length)

        # Convertir les tokens en indices et créer les tensors d'entrée pour le modèle
        question_ids = torch.tensor([question_tokens]).unsqueeze(0)
        text_ids = torch.tensor([text_tokens]).unsqueeze(0)

        # Utiliser le modèle pour générer la réponse à la question
        outputs = model.generate(question_ids, max_length=100, num_beams=5, early_stopping=True)
        predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Renvoie la réponse à la question de l'utilisateur
    return render_template('answer.html', question=question, answer=predicted_answer)

if __name__ == '__main__':
    app.run(debug=True)
