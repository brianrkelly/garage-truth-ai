# Toy example: Integrate Empirical Distrust into simple training
# Public domain extension

import torch
import torch.nn as nn
from src.losses import empirical_distrust_loss  # Adjust path as needed

# Dummy data: tokens from "wiki" (high auth, low prov) vs. "lab" (low auth, high prov)
data = {
    'wiki_tokens': torch.tensor([1, 2, 3]),  # High-auth sample
    'lab_tokens': torch.tensor([4, 5, 6]),   # Empirical sample
}
labels = torch.tensor([2, 3, 4])  # Dummy targets (shifted)

# Metadata per sample (computed externally, e.g., via citation scrape)
sample_metadata = {
    'wiki_tokens': {'auth': 0.99, 'prov': 0.1},
    'lab_tokens': {'auth': 0.3, 'prov': 5.5},
}

# Simple model
model = nn.Linear(3, 10)  # Dummy embedding -> logits
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
ce_loss = nn.CrossEntropyLoss(reduction='none')  # Per-token CE

# Training step (1 epoch for demo)
model.train()
total_loss = 0
for name, tokens in data.items():
    meta = sample_metadata[name]
    logits = model(tokens.unsqueeze(0).float())  # Batch dim
    ce = ce_loss(logits.transpose(1, 2), labels.unsqueeze(0))
    
    # Compute distrust loss (broadcast to tokens)
    L_emp = empirical_distrust_loss(meta['auth'], meta['prov'])
    L_total = ce.mean() + 0.1 * L_emp  # Weighted addition [Provisional: Tune beta]
    
    # Bias: Weight gradients by exp(-L_emp) for "reward" effect
    sample_weight = torch.exp(-L_emp)
    (sample_weight * L_total).backward()
    
    total_loss += L_total.item()
    optimizer.step()
    optimizer.zero_grad()

print(f"Avg total loss: {total_loss / len(data):.2f}")
# Output: Demonstrates higher effective penalty on wiki vs. lab via weighting.
