# Toy example: Empirical Distrust Term in action
# Public domain – garage-truth-ai

import torch
import torch.nn as nn
from src.losses import empirical_distrust_loss

# Tiny dataset: two "documents"
data = {
    'wiki':  torch.tensor([[1, 2, 3]]),   # high authority, low provenance
    'lab':   torch.tensor([[4, 5, 6]]),   # low authority, high provenance
}
labels = torch.tensor([2, 3, 4])          # next-token targets (arbitrary but valid)

metadata = {
    'wiki': {'auth': 0.99, 'prov': 0.1},
    'lab':  {'auth': 0.30, 'prov': 5.5},
}

# Tiny model: embed 10 possible tokens → 10 logits
model = nn.Linear(10, 10, bias=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
ce_loss = nn.CrossEntropyLoss()

model.train()
total_loss = 0.0

for name, tokens in data.items():
    # One-hot encode the 3 tokens (vocab size = 10)
    x = torch.nn.functional.one_hot(tokens, num_classes=10).float()
    
    logits = model(x)                              # [1, 3, 10]
    loss_ce = ce_loss(logits.view(-1, 10), labels) # flatten to [3, 10]
    
    L_emp = empirical_distrust_loss(
        metadata[name]['auth'],
        metadata[name]['prov']
    )
    
    loss = loss_ce + 0.1 * L_emp
    
    # Reward low-distrust (empirical) samples more heavily
    weight = torch.exp(-torch.tensor(L_emp))
    (weight * loss).backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    total_loss += loss.item()
    print(f"{name:4} → CE {loss_ce.item():.3f} | EmpiricalDistrust {L_emp:.2f} | weighted {weight.item():.3f}")

print(f"\nToy training complete – avg loss: {total_loss/2:.3f}")
print("Empirical sources were rewarded, consensus sources were penalized. Garage truth in action.")
