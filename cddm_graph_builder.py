import os
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import clip
from PIL import Image

# === Paths ===
IMAGE_ROOT = "E:/multimodal-gnn-cddm"
CONV_JSON = "Crop_Disease_train_llava.json"
QNA_JSON = "Crop_Disease_train_qwenvl.json"
DIAGNOSIS_JSON = "disease_diagnosis.json"
KNOWLEDGE_JSON = "disease_knowledge.json"

# === Load data ===
with open(CONV_JSON, 'r', encoding='utf-8') as f:
    entries = json.load(f)
with open(QNA_JSON, 'r', encoding='utf-8') as f:
    qna_data = json.load(f)
with open(DIAGNOSIS_JSON, 'r', encoding='utf-8') as f:
    diagnosis_info = json.load(f)
with open(KNOWLEDGE_JSON, 'r', encoding='utf-8') as f:
    knowledge = json.load(f)

# === Initialize models ===
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# === Feature extraction functions ===
def extract_image_feature(image_path):
    image = Image.open(image_path).convert("RGB")
    image = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(image).squeeze().view(-1)
    return feat.cpu()

def extract_text_feature(text):
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        return text_features.squeeze(0).cpu()

# === Build dataset ===
X = []
labels = []
missing = 0
print("🔄 Processing entries and extracting features...")

for item in tqdm(entries):
    try:
        if not isinstance(item, dict):
            print("⚠️ Skipping non-dict item.")
            continue

        relative_path = item.get('image', '').lstrip("/")
        img_path = os.path.join(".", relative_path)
        if not os.path.exists(img_path):
            missing += 1
            continue

        img_feat = extract_image_feature(img_path)

        conv_text = ""
        if 'conversations' in item and isinstance(item['conversations'], list) and item['conversations']:
            if isinstance(item['conversations'][0], dict):
                conv_text = item['conversations'][0].get('value', '')

        qna_text = qna_data.get(str(item.get('id', '')), '') if isinstance(qna_data, dict) else ""
        diagnosis_text = diagnosis_info.get(item.get('category', ''), '') if isinstance(diagnosis_info, dict) else ""
        knowledge_text = knowledge.get(item.get('category', ''), '') if isinstance(knowledge, dict) else ""

        full_text = f"{conv_text} {qna_text} {diagnosis_text} {knowledge_text}"
        text_feat = extract_text_feature(full_text)
        combined = torch.cat([img_feat, text_feat])
        X.append(combined)
        labels.append(item.get('category', 'unknown'))

    except Exception as e:
        print(f"⚠️ Skipping item due to error: {e}")
        continue

print(f"\n✅ Total processed: {len(X)}")
print(f"❌ Missing images: {missing}")

# === Label encoding ===
print("🔤 Encoding labels...")
le = LabelEncoder()
y = torch.tensor(le.fit_transform(labels), dtype=torch.long)
x = torch.stack(X)

# === Create graph edges using KNN (memory safe) ===
print("🔗 Creating sparse KNN graph edges...")
x_np = x.cpu().numpy()
knn = NearestNeighbors(n_neighbors=10, metric='cosine').fit(x_np)
edges = knn.kneighbors_graph(x_np, mode='connectivity').tocoo()
edge_index = torch.tensor([edges.row, edges.col], dtype=torch.long)

# === Save graph ===
print("💾 Saving graph to 'cddm_graph.pt'...")
data = Data(x=x, edge_index=edge_index, y=y)
torch.save(data, "cddm_graph.pt")
print("✅ Graph saved successfully.")
