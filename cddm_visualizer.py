# cddm_visualizer.py

"""
Visualize learned GNN node embeddings using t-SNE or PCA
and plot training loss/accuracy vs epochs from a log file.
"""

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

from cddm_gnn_trainer import GCN

# -------- CONFIG -------- #
GRAPH_PATH = "./output/graph_data.pt"
MODEL_PATH = "./output/gnn_model.pth"
LOG_PATH = "./output/training_log.txt"
VIS_TYPE = "tsne"  # or "pca"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD GRAPH AND MODEL -------- #
print("Loading graph and model...")
graph = torch.load(GRAPH_PATH)
model = GCN(graph.num_node_features, 128, graph.y.max().item() + 1)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

with torch.no_grad():
    out = model(graph.x.to(device), graph.edge_index.to(device))
    embeddings = out.cpu().numpy()
    labels = graph.y.cpu().numpy()

# -------- DIMENSION REDUCTION -------- #
if VIS_TYPE == "tsne":
    print("Reducing dimensions using t-SNE...")
    reduced = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)
else:
    print("Reducing dimensions using PCA...")
    reduced = PCA(n_components=2).fit_transform(embeddings)

# -------- PLOT NODE EMBEDDINGS -------- #
plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=20)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title(f"{VIS_TYPE.upper()} of GNN Node Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("./output/embedding_visualization.png")
plt.show()
print("Node embedding visualization saved as ./output/embedding_visualization.png")

# -------- OPTIONAL: PLOT LOSS AND ACCURACY -------- #
if os.path.exists(LOG_PATH):
    print("Plotting training loss and accuracy...")
    epochs, losses, accs = [], [], []
    with open(LOG_PATH, 'r') as f:
        for line in f:
            if line.startswith("Epoch"):
                parts = line.strip().split()
                epochs.append(int(parts[1]))
                losses.append(float(parts[3]))
                accs.append(float(parts[6]))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(epochs, losses, 'r-', label='Loss')
    ax2.plot(epochs, accs, 'b-', label='Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='r')
    ax2.set_ylabel('Accuracy', color='b')
    plt.title("Training Loss and Accuracy over Epochs")
    fig.tight_layout()
    plt.savefig("./output/loss_accuracy_plot.png")
    plt.show()
    print("Loss/accuracy plot saved as ./output/loss_accuracy_plot.png")
else:
    print("No training log found at ./output/training_log.txt")
