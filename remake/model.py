from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithLMHead

# Charger le tokenizer et le modèle pour BERT
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model_bert.eval()

# Charger le tokenizer et le modèle pour RoBERTa
tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
model_roberta = AutoModelForSequenceClassification.from_pretrained("roberta-base")
model_roberta.eval()

# Charger le tokenizer et le modèle pour DistilBERT
tokenizer_distilbert = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_distilbert = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model_distilbert.eval()

# Charger le tokenizer et le modèle pour XLNet
tokenizer_xlnet = AutoTokenizer.from_pretrained("xlnet-base-cased")
model_xlnet = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased")
model_xlnet.eval()

# Charger le tokenizer et le modèle pour ELECTRA
tokenizer_electra = AutoTokenizer.from_pretrained("electra-base-discriminator")
model_electra = AutoModelForSequenceClassification.from_pretrained("electra-base-discriminator")
model_electra.eval()

# Charger le tokenizer et le modèle pour GPT
tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2")
model_gpt = AutoModelWithLMHead.from_pretrained("gpt2")
model_gpt.eval()

# Charger le tokenizer et le modèle pour GPT-2
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2-medium")
model_gpt2 = AutoModelWithLMHead.from_pretrained("gpt2-medium")
model_gpt2.eval()

# Charger le tokenizer et le modèle pour Mistral
tokenizer_mistral = AutoTokenizer.from_pretrained("MistralAI/mBART-en-fr")
model_mistral = AutoModelWithLMHead.from_pretrained("MistralAI/mBART-en-fr")
model_mistral.eval()

# Charger le tokenizer et le modèle pour Llama
tokenizer_llama = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model_llama = AutoModelWithLMHead.from_pretrained("facebook/bart-large-cnn")
model_llama.eval()
