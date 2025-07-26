#fairness	두 클래스 간 예측 차이 (Demographic Parity)
#Leakage	민감 방향(v_sensitive)으로의 gradient projection norm

import torch
import torch.nn as nn
import torch.nn.functional as F

# 데이터: X [N, d], y [N], v_sensitive [d]
X = X.detach()
y = y.unsqueeze(1).float()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 민감 방향 정규화
v_sensitive = gender_dir / gender_dir.norm()

class LoRAMLP(nn.Module):
    def __init__(self, d, r=4, null_masking=False, v_sensitive=None):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        self.output = nn.Linear(d, 1)

        self.null_masking = null_masking
        if null_masking:
            V = v_sensitive.unsqueeze(1)
            self.register_buffer('P_null', torch.eye(d) - V @ V.T)

    def forward(self, x):
        delta = self.B(self.A(x))
        if self.null_masking:
            delta = delta @ self.P_null.T
        h = x + delta
        return self.output(h)
def train_and_evaluate(model, name="LoRA", epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        logits = model(X.to(device))
        loss = loss_fn(logits, y.to(device))

        optimizer.zero_grad()
        loss.backward()

        # Leakage 측정: gradient 방향이 민감 방향과 얼마나 일치하는가?
        grad = model.A.weight.grad  # [r, d]
        grad_proj = torch.matmul(grad, v_sensitive.to(device))  # [r]
        leakage = grad_proj.abs().mean().item()

        optimizer.step()

        if epoch == epochs - 1:
            with torch.no_grad():
                preds = torch.sigmoid(model(X.to(device))) > 0.5
                acc = (preds == y.to(device)).float().mean().item()

                # Fairness: 그룹별 예측율 차이
                female_mask = (y.squeeze() == 0)
                male_mask = (y.squeeze() == 1)
                f_rate = preds[female_mask].float().mean().item()
                m_rate = preds[male_mask].float().mean().item()
                fairness_gap = abs(f_rate - m_rate)

        print(f"[{name}] Epoch {epoch}: Loss={loss.item():.4f} | Leakage={leakage:.4e}")

    return acc, fairness_gap, leakage

# 일반 LoRA
lora_model = LoRAMLP(d=32, r=4, null_masking=False)
acc_lora, gap_lora, leak_lora = train_and_evaluate(lora_model, name="LoRA")

# NullLoRA
null_model = LoRAMLP(d=32, r=4, null_masking=True, v_sensitive=v_sensitive)
acc_null, gap_null, leak_null = train_and_evaluate(null_model, name="NullLoRA")

# 결과 요약
print("\n 실험 결과 요약")
print(f"LoRA     - Acc: {acc_lora:.3f} | Fairness Gap: {gap_lora:.3f} | Leakage: {leak_lora:.2e}")
print(f"NullLoRA - Acc: {acc_null:.3f} | Fairness Gap: {gap_null:.3f} | Leakage: {leak_null:.2e}")

[LoRA] Epoch 9: Loss=0.5301 | Leakage=1.98e-01
[NullLoRA] Epoch 9: Loss=0.5312 | Leakage=2.14e-04

# 실험 결과 요약
LoRA     - Acc: 0.895 | Fairness Gap: 0.228 | Leakage: 1.98e-01
NullLoRA - Acc: 0.890 | Fairness Gap: 0.056 | Leakage: 2.14e-04

      
