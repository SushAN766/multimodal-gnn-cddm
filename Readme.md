# Analysis of Multimodal Data Using GNN (CDDM Dataset)

This project demonstrates how to process multimodal data (images + text) from the **CDDM dataset**, convert it into graph format, and train a **Graph Neural Network (GNN)** to perform classification using PyTorch Geometric.

---

**Refer to [CDDMBench](https://github.com/SushAN766/CDDMBench) for more details.**

---
## CDDM dataset
The CDDM dataset includes images and conversation data. 
### CDDM images:
Please download CDDM images from the following link and extract it to the /dataset/ directory.
- [Google Drive](https://drive.google.com/file/d/1kfB3zkittoef4BasOhwvAb8Cb66EPXst/view?usp=sharing)

---

### CDDM conversation:
We offer the conversation data in two formats suitable for training Qwen-VL and LLaVA models. The data covers crop disease diagnosis and knowledge.

Please extract the conversation data to the /dataset/VQA/ directory. 
- [Qwen-VL training data](VQA/Crop_Disease_train_qwenvl.zip)
- [LLaVA training data](VQA/Crop_Disease_train_llava.zip)
- [Test data](VQA/test_dataset.zip)

---

## Paper
For more details, please refer to our paper: [ECCV 2024 Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11606.pdf)  , [arxiv](https://arxiv.org/abs/2503.06973)

--- 

## 📁 Folder Structure
```plaintext
multimodal-gnn-cddm/                   
├── dataset/images/                    # CDDM image files
├── gnn-env/                           # Python virtual environment (excluded via .gitignore)
├── VQA/                               # Visual Question Answering components
├── .gitignore                         # Git ignore rules
├── Crop_Disease_train_llava.json     # CDDM conversation file (LLaVA format)
├── Crop_Disease_train_qwenvl.json    # CDDM conversation file (Qwen-VL format)
├── disease_knowledge.json            # Domain-specific knowledge base
├── disease_diagnosis.json            # Ground-truth diagnosis labels
├── cddm_graph_builder.py             # Feature extraction + graph construction script
├── cddm_gnn_trainer.py               # GCN model training script
├── cddm_visualizer.py                # Training & embedding visualization (PCA/t-SNE)
├── output/                           # Output directory (auto-created)
│   ├── ccdm_graph.pt                 # Saved PyG graph
│   ├── gnn_model.pth                 # Trained model weights
│   ├── training_log.txt              # Training logs (optional)
│   ├── embedding_visualization.png  # Visual representation of embeddings
│   └── loss_accuracy_plot.png       # Training loss & accuracy plot
└── README.md                         # Project documentation


```
---

## 🔧 Setup Instructions

### ✅ Step 1: Create Virtual Environment (Optional)

```bash
python -m venv gnn-env
source gnn-env/bin/activate        # Linux/macOS
gnn-env\Scripts\activate          # Windows
```
## ✅ Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+cu117.html  # Replace cu117 if needed
pip install transformers
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install git+https://github.com/openai/CLIP.git
```

## 🚀 How to Run

### 📌 Step 1: Graph Construction

```shell
python cddm_graph_builder.py
```

- Loads Crop_Disease_train_llava.json and image folder
- Extracts image features using ResNet50
- Extracts text features using BERT
- Constructs a graph and saves it as output/graph_data.pt

### 📌 Step 2: Train GCN Model

```shell
python cddm_gnn_trainer.py
```
- Loads the graph
- Trains a 2-layer GCN for classification
- Saves model as output/gnn_model.pth

### 📌 Step 3: Visualize Results

```shell
python cddm_visualizer.py
```
- Reduces node embeddings using t-SNE or PCA
- Generates: embedding_visualization.png
- If training_log.txt exists, also plots loss_accuracy_plot.png

## 🎯 Project Objectives

- Preprocess multimodal data (image + text)  
- Convert preprocessed data into graph representation  
- Apply GNN models for prediction and insight  

---

## 📈 Possible Improvements

- Use **GraphSAGE** or **GAT** instead of GCN  
- Construct edges based on **cosine similarity** or **KNN**  
- Use **multi-task** or **multi-label** outputs  

---

## 📬 Credits

- **Dataset**: Crop Disease Diagnosis Multimodal (CDDM)  
- **Libraries**: PyTorch Geometric,HuggingFace Transformers,scikit-learn,matplotlib
