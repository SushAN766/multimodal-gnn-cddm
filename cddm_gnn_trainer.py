# cddm_gnn_trainer.py

"""
Train a GNN model (GraphSAGE) on the graph created from the CDDM multimodal dataset.
This fulfills Objective 3 of your project.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import os

# -------- LOAD GRAPH -------- #
GRAPH_PATH = "cddm_graph.pt"
print(f"Loading graph from {GRAPH_PATH}...")
graph = torch.load(GRAPH_PATH, weights_only=False)
graph = graph.to('cuda' if torch.cuda.is_available() else 'cpu')

# -------- CHECK OR CREATE MASKS -------- #
if not hasattr(graph, 'train_mask') or not hasattr(graph, 'test_mask'):
    print("⚠️ train_mask or test_mask not found. Creating them now...")
    indices = list(range(graph.num_nodes))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=graph.y.cpu())

    graph.train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.x.device)
    graph.test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.x.device)
    graph.train_mask[train_idx] = True
    graph.test_mask[test_idx] = True

# -------- DEBUG: Check Class Balance -------- #
print("Classes:", torch.unique(graph.y))
print("Train set distribution:", torch.bincount(graph.y[graph.train_mask]))
print("Test set distribution:", torch.bincount(graph.y[graph.test_mask]))

# -------- DEFINE GraphSAGE MODEL -------- #
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# -------- TRAINING SETUP -------- #
device = graph.x.device
model = GraphSAGE(graph.num_node_features, 128, graph.y.max().item() + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------- TRAIN LOOP -------- #
print("Starting training...")
for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    out = model(graph.x, graph.edge_index)
    loss = F.cross_entropy(out[graph.train_mask], graph.y[graph.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    pred = out.argmax(dim=1)
    correct = pred[graph.test_mask] == graph.y[graph.test_mask]
    acc = int(correct.sum()) / int(graph.test_mask.sum())

    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

# -------- SAVE MODEL -------- #
os.makedirs("./output", exist_ok=True)
torch.save(model.state_dict(), "./output/gnn_model.pth")
print("Model training complete and saved to ./output/gnn_model.pth")
