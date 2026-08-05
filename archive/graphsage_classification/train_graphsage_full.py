import os

import matplotlib
matplotlib.use("Agg")  # no display available on HPC compute nodes
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch_geometric.nn import SAGEConv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
wandb.login(key=os.environ["WANDB_KEY"])

DATA_PATH = os.path.join(SCRIPT_DIR, "full_graph_data.pt")
data = torch.load(DATA_PATH, weights_only=False)
print(data)

num_features = data.num_node_features   # 128
num_classes  = int(data.y.max()) + 1    # 40

print(f"\nFeatures : {num_features}")
print(f"Classes  : {num_classes}")
print(f"Train    : {data.train_mask.sum().item():,}")
print(f"Val      : {data.val_mask.sum().item():,}")
print(f"Test     : {data.test_mask.sum().item():,}")


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels,     hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x                          # node embeddings after message passing

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)         # logits, shape (N, num_classes)


HIDDEN  = 256
DROPOUT = 0.5
LR      = 1e-3
EPOCHS  = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

run = wandb.init(
    project="graphsage-ogbn-arxiv",
    name="full-graph-training",
    config={
        "dataset": "full_graph_data.pt",
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "hidden_channels": HIDDEN,
        "dropout": DROPOUT,
        "lr": LR,
        "epochs": EPOCHS,
        "weight_decay": 5e-4,
        "device": str(device),
    },
)

model = GraphSAGE(
    in_channels=num_features,
    hidden_channels=HIDDEN,
    out_channels=num_classes,
    dropout=DROPOUT,
).to(device)

data = data.to(device)

print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=20
)


@torch.no_grad()
def evaluate():
    model.eval()
    logits = model(data.x, data.edge_index)
    preds  = logits.argmax(dim=1)

    def acc(mask):
        return (preds[mask] == data.y[mask]).float().mean().item()

    return acc(data.train_mask), acc(data.val_mask), acc(data.test_mask)


history = {"train_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
best_val, best_test = 0.0, 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss   = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    train_acc, val_acc, test_acc = evaluate()
    scheduler.step(val_acc)

    history["train_loss"].append(loss.item())
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["test_acc"].append(test_acc)

    wandb.log({
        "train_loss": loss.item(),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "lr": optimizer.param_groups[0]["lr"],
    }, step=epoch)

    if val_acc > best_val:
        best_val  = val_acc
        best_test = test_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:>3}  loss={loss.item():.4f}  "
              f"train={train_acc:.4f}  val={val_acc:.4f}  test={test_acc:.4f}")

print(f"\nBest  val={best_val:.4f}  test={best_test:.4f}")
wandb.run.summary["best_val_acc"] = best_val
wandb.run.summary["best_test_acc"] = best_test


epochs = range(1, EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(epochs, history["train_loss"])
ax1.set_title("Cross-entropy loss (train)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

ax2.plot(epochs, history["train_acc"], label="train")
ax2.plot(epochs, history["val_acc"],   label="val")
ax2.plot(epochs, history["test_acc"],  label="test")
best_epoch = history["val_acc"].index(best_val) + 1
ax2.axvline(best_epoch, color="grey", linestyle="--", linewidth=0.8, label="best val")
ax2.set_title("Node classification accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, "full_graph_training_curves.png")
plt.savefig(plot_path, dpi=150)
wandb.log({"training_curves": wandb.Image(plot_path)})

print(f"Final result  —  val: {best_val:.4f}  |  test: {best_test:.4f}")


model.load_state_dict(best_state)
model.to(device)
model.eval()

with torch.no_grad():
    # Node embeddings after message passing, before the classifier head
    embeddings = model.encode(data.x, data.edge_index).cpu()  # (num_nodes, HIDDEN)

print(f"Embedding matrix: {tuple(embeddings.shape)}")
print("  embedding[i] corresponds to original node_id = data.original_node_ids[i]")


weights_path = os.path.join(SCRIPT_DIR, "full_graph_weights.pt")
embeddings_path = os.path.join(SCRIPT_DIR, "full_graph_embeddings.pt")

torch.save(best_state, weights_path)
torch.save(embeddings, embeddings_path)   # (num_nodes, HIDDEN)

print("Saved:")
print("  full_graph_weights.pt    — model state dict (best val checkpoint)")
print("  full_graph_embeddings.pt — post-SAGE node embeddings")

wandb.finish()
