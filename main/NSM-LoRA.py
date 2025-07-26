import torch.nn as nn

class NullMaskedLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, sensitive_dir=None):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        
        self.A = nn.Parameter(torch.randn(out_features, r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.sensitive_dir = sensitive_dir  # shape: (in_features,)

    def forward(self, x):
        delta = self.A @ self.B  # (out, in)
        if self.sensitive_dir is not None:
            v = self.sensitive_dir.view(-1, 1)  # (in, 1)
            P = torch.eye(self.in_features, device=x.device) - v @ v.T
            delta = delta @ P  # Project out sensitive direction
        return x @ delta.T
        
class AdapterModel(nn.Module):
    def __init__(self, base_model, sensitive_basis):
        super().__init__()
        self.base = base_model
        self.null_lora = NullSpaceLoRALinear(768, 768, r=8, V_sensitive=sensitive_basis)

    def forward(self, x):
        h = self.base(x)
        h += self.null_lora(h)  # residual update in null-space
        return h
