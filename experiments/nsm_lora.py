import torch
import torch.nn as nn
from peft.tuners.lora import Linear as LoraLinear

class NullSpaceLoraLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, sensitive_direction: torch.Tensor):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r

        self.A = nn.Linear(self.in_features, r, bias=False)
        self.B = nn.Linear(r, self.out_features, bias=False)

        # 민감 방향 V (normalized)
        V = sensitive_direction.unsqueeze(1)  # [d, 1]
        self.register_buffer('P_null', torch.eye(self.out_features) - V @ V.T)

    def forward(self, x):
        lora_out = self.B(self.A(x))  # LoRA output
        lora_out = lora_out @ self.P_null.T  # Null-space projection
        return lora_out
from transformers import DistilBertModel, DistilBertConfig

class DistilBertWithNullLora(nn.Module):
    def __init__(self, r=4, sensitive_vector=None):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 1)

        # LoRA 적용할 위치: Transformer block의 FFN
        for name, module in self.bert.transformer.layer[0].ffn.named_children():
            if isinstance(module, nn.Linear) and module.out_features == 768:
                print(f"Replacing module: {name}")
                setattr(self.bert.transformer.layer[0].ffn, name,
                        NullSpaceLoraLinear(module, r=r, sensitive_direction=sensitive_vector))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

###PEFT 없이 synthetic 데이터를 DistilBERT에 넣기
# synthetic 데이터
X = X.detach()  # shape: [200, d]
y = y.detach().unsqueeze(1).float()

# 민감 방향 (예: SVD/PCA로 추정된 gender 방향)
v_sensitive = gender_dir  # shape: [d]

# 모델 선언
model = DistilBertWithNullLora(r=4, sensitive_vector=v_sensitive)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
loss_fn = nn.BCEWithLogitsLoss()

# 임시 훈련 루프
for epoch in range(5):
    model.train()
    logits = model.bert(inputs_embeds=X.unsqueeze(1)).last_hidden_state[:, 0]
    pred = model.classifier(logits)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
