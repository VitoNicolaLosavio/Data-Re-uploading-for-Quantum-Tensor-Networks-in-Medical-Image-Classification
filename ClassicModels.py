from torch.utils.data import TensorDataset, DataLoader
from itertools import combinations
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import random

torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_value = 12345
np.random.seed(seed_value)
random.seed(seed_value)


class TripletAutoencoder(nn.Module):
    def __init__(self, input_dim=784, bottleneck_dim=8):
        super(TripletAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed


# Function for build triplets
def generate_triplets(labels):
    triplets = []
    labels = labels.cpu().numpy()

    for label in set(labels):
        pos_idx = [i for i in range(len(labels)) if labels[i] == label]
        neg_idx = [i for i in range(len(labels)) if labels[i] != label]
        if len(pos_idx) < 2:
            continue
        for anchor, positive in combinations(pos_idx, 2):
            negative = random.choice(neg_idx)
            triplets.append((anchor, positive, negative))
    return triplets


# === Training ===
def train_triplet_autoencoder(
        model, X, y,
        n_epochs=100, batch_size=32, lr=1e-3,
        margin=1.0, alpha=0.5, verbose=False
):

    model.to(device)
    criterion_recon = nn.MSELoss()
    criterion_triplet = nn.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            emb, xb_recon = model(xb)

            loss_recon = criterion_recon(xb_recon, xb)

            # Triplet loss
            triplets = generate_triplets(yb)
            if triplets:
                anchor = torch.stack([emb[a] for a, _, _ in triplets])
                positive = torch.stack([emb[p] for _, p, _ in triplets])
                negative = torch.stack([emb[n] for _, _, n in triplets])
                loss_triplet = criterion_triplet(anchor, positive, negative)
                loss = alpha * loss_recon + loss_triplet
            else:
                loss = loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss:.4f}")

    model.to('cpu')
    return model


# Function for extract the embeddings
def extract_embeddings(model, data):
    model.to(device)
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        encoded = model.encoder(data_tensor)
        return encoded.cpu().numpy()
