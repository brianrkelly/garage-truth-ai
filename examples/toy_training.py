# Toy example: Integrate Empirical Distrust into simple training
# Public domain extension

import torch
import torch.nn as nn
from src.losses import empirical_distrust_loss

# Dummy data
data = {
    'wiki_tokens': torch.tensor([1, 2, 3]),
    'lab_tokens': torch.tensor([4, 5, 6]),
}
labels = torch.tensor([2, 3, 4])

sample_metadata = {
    'wiki_tokens': {'auth': 0.99, 'prov': 0.1},
    'lab_tokens': {'auth': 0.3, 'prov': 5.5},
}

model = nn.Linear(3, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
ce_loss = nn.CrossEntropyLoss(reduction='none')

model.train()
total_loss = 0.0
for name, tokens in data.items():
    meta = sample_metadata[name]
    logits = model(tokens.unsqueeze(0).float())
    ce = ce_loss(logits.transpose(1, 2), labels.unsqueeze(0))
    
    L_emp = empirical_distrust_loss(meta['auth'], meta['prov'])
    L_total = ce.mean() + 0.1 * L_emp
    
    sample_weight = torch.exp(-L_emp)
    (sample_weight * L_total).backward()
    
    optimizer.step()
    optimizer.zero_grad()
    total_loss += L_total.item()

print(f"Toy training complete – avg loss: {total_loss/len(data):.2f}")
