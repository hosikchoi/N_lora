import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD

def compute_sensitive_direction(embeddings, labels, sensitive_value):
    """
    embeddings: (n_samples, d) torch.Tensor
    labels: (n_samples,) list or array of sensitive attribute values
    sensitive_value: value of sensitive attribute (e.g., 'male')
    
    Returns: direction vector v (d,) torch.Tensor
    """
    X = embeddings[labels == sensitive_value]
    X_centered = X - X.mean(dim=0, keepdim=True)
    svd = TruncatedSVD(n_components=1)
    v = svd.fit(X_centered.cpu().numpy()).components_[0]  # shape (d,)
    return torch.tensor(v, dtype=torch.float32, device=embeddings.device)
