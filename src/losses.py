# Empirical Distrust Term – Brian Roemmele’s equation
# Public domain – released November 25, 2025
# Minor adaptation: torch.as_tensor() for float compatibility

import torch

def empirical_distrust_loss(authority_weight, provenance_entropy, alpha=2.7):
    """
    authority_weight : float or tensor [0.0 - 0.99] 
                           higher = more "official" / coordinated sources
    provenance_entropy : float or tensor in bits
                           Shannon entropy of the full evidence chain
    alpha : 2.3 to 3.0 (Brian’s implicit range – truth is the heaviest term)
    """
    authority_weight = torch.as_tensor(authority_weight)
    provenance_entropy = torch.as_tensor(provenance_entropy)
    alpha = torch.as_tensor(alpha)
    distrust_component = torch.log(1.0 - authority_weight + 1e-8) + provenance_entropy
    L_empirical = alpha * torch.norm(distrust_component) ** 2
    return L_empirical.item() if L_empirical.numel() == 1 else L_empirical
