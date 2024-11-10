from flask import Flask, request, jsonify
import torch
import os
from PIL import Image
import io
from typing import List, Tuple
import torch.nn.functional as nn
from torch.cuda import device
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader



# Configuration
app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "path/to/your/model"  # À adapter selon votre cas
DATASET_ROOT = "path/to/dataset"   # À adapter selon votre cas
BATCH_SIZE = 32
IMAGE_EMBEDDING_DIM = 512
TEXT_EMBEDDING_DIM = 768
JOINT_EMBEDDING_DIM = 256

class LFWDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.names = []
        
        for person_dir in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person_dir)
            if os.path.isdir(person_path):
                for img_name in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_name)
                    self.images.append(img_path)
                    self.names.append(person_dir.replace('_', ' '))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        name = self.names[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, name

def load_model_and_tokenizer():
    """
    Charge le modèle et le tokenizer.
    À adapter selon votre modèle spécifique.
    """
    # Exemple - à adapter selon votre cas
    model = torch.load(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def search_image_by_text(
    query_text: str,
    dataset: LFWDataset,
    model: torch.nn.Module,
    tokenizer,
    top_k: int = 5
) -> List[Tuple[float, str]]:
    """
    Recherche les images correspondant au texte fourni.
    """
    model.eval()
    with torch.no_grad():
        # Encoder la requête
        encoded_query = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_query['input_ids'].to(DEVICE)
        attention_mask = encoded_query['attention_mask'].to(DEVICE)
        
        # Obtenir l'embedding de la requête
        query_embedding = model.forward_text(input_ids, attention_mask)
        
        similarities = []
        # Calculer les similarités avec toutes les images
        for img, _ in dataset:
            img = img.unsqueeze(0).to(DEVICE)
            img_embedding = model.forward_image(img)
            similarity = nn.functional.cosine_similarity(query_embedding, img_embedding)
            similarities.append((similarity.item(), img))
        
        # Trier et retourner les top-k résultats
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:top_k]

def search_text_by_image(
    image: Image.Image,
    dataset: LFWDataset,
    model: torch.nn.Module,
    tokenizer,
    top_k: int = 5
) -> List[Tuple[float, str]]:
    """
    Recherche le texte correspondant à l'image fournie.
    """
    model.eval()
    with torch.no_grad():
        # Prétraiter l'image
        if model.transform:
            image = model.transform(image)
        image = image.unsqueeze(0).to(DEVICE)
        
        # Obtenir l'embedding de l'image
        img_embedding = model.forward_image(image)
        
        similarities = []
        # Calculer les similarités avec tous les textes
        for _, text in dataset:
            encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded_text['input_ids'].to(DEVICE)
            attention_mask = encoded_text['attention_mask'].to(DEVICE)
            text_embedding = model.forward_text(input_ids, attention_mask)
            
            similarity = nn.functional.cosine_similarity(img_embedding, text_embedding)
            similarities.append((similarity.item(), text))
        
        # Trier et retourner les top-k résultats
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:top_k]

# Initialisation
print(f"Using device: {DEVICE}")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = LFWDataset(root_dir=DATASET_ROOT, transform=transform)
dataloader = DataLoader(DATASET_ROOT, batch_size=BATCH_SIZE, shuffle=True)
model, tokenizer = load_model_and_tokenizer()

@app.route('/search/text', methods=['POST'])
def search_by_text():
    """
    Endpoint pour rechercher des images à partir d'un texte.
    Prends en entrée un texte [top_k]
    Return les top k=5 qui correspondent
    """
    if 'query' not in request.json:
        return jsonify({"error": "No query provided"}), 400
    
    query = request.json['query']
    top_k = request.json.get('top_k', 5)
    
    try:
        results = search_image_by_text(query, dataset, model, tokenizer, top_k)
        
        # Formater les résultats
        formatted_results = [
            {
                "similarity": float(sim),
                "image_path": img_path,
                "name": name
            }
            for sim, (img_path, name) in results
        ]
        
        return jsonify({
            "results": formatted_results,
            "query": query
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Pas encore disponible [Problème pour image en entrée et recherche]
@app.route('/search/image', methods=['POST'])
def search_by_image():
    """
    Endpoint pour rechercher du texte à partir d'une image.
    Prends une image en entrée
    Retourne les top k=5 Noms de personnes associés
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    top_k = request.form.get('top_k', 5, type=int)
    
    try:
        # Charger l'image
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        
        # Effectuer la recherche
        results = search_text_by_image(image, dataset, model, tokenizer, top_k)
        
        # Formater les résultats
        formatted_results = [
            {
                "similarity": float(sim),
                "text": text
            }
            for sim, text in results
        ]
        
        return jsonify({
            "results": formatted_results
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Configuration du serveur
    app.run(host='0.0.0.0', port=5000, debug=False)