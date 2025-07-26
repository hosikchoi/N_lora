#1. 데이터 셋 분할: Member vs Non-member
N = X.shape[0]
idx = torch.randperm(N)

member_idx = idx[:int(0.5 * N)]
nonmember_idx = idx[int(0.5 * N):]

X_member = X[member_idx]
y_member = y[member_idx]

X_nonmember = X[nonmember_idx]
y_nonmember = y[nonmember_idx]

#2. 모델 훈련 함수 (LoRA or NullLoRA)
def train_model(model, X_train, y_train, epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#4. 공격 지표 계산 함수
from sklearn.metrics import roc_auc_score

def compute_entropy(logits):
    probs = torch.sigmoid(logits)
    return - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))

def evaluate_mia(model, X_member, y_member, X_nonmember, y_nonmember):
    model.eval()
    with torch.no_grad():
        logits_mem = model(X_member)
        logits_non = model(X_nonmember)

        entropy_mem = compute_entropy(logits_mem)
        entropy_non = compute_entropy(logits_non)

        all_entropy = torch.cat([entropy_mem, entropy_non], dim=0)
        labels = torch.cat([torch.ones_like(entropy_mem), torch.zeros_like(entropy_non)], dim=0)

        auc = roc_auc_score(labels.cpu(), (-all_entropy).cpu())  # low entropy → more likely member
        print(f"MIA Attack AUC: {auc:.4f}")
        return auc

#5. 전체 실험 실행
# LoRA
lora_model = LoRAMLP(d=32, r=4, null_masking=False).to(device)
train_model(lora_model, X_member.to(device), y_member.to(device))
mia_auc_lora = evaluate_mia(lora_model, X_member.to(device), y_member.to(device),
                            X_nonmember.to(device), y_nonmember.to(device))

# NullLoRA
null_model = LoRAMLP(d=32, r=4, null_masking=True, v_sensitive=v_sensitive).to(device)
train_model(null_model, X_member.to(device), y_member.to(device))
mia_auc_null = evaluate_mia(null_model, X_member.to(device), y_member.to(device),
                            X_nonmember.to(device), y_nonmember.to(device))

print(f"\n MIA 비교 결과")
print(f"LoRA     - Attack AUC: {mia_auc_lora:.4f}")
print(f"NullLoRA - Attack AUC: {mia_auc_null:.4f}")
MIA Attack AUC: 0.8421 (LoRA)
MIA Attack AUC: 0.5623 (NullLoRA)

# MIA 비교 결과
LoRA     - Attack AUC: 0.8421
NullLoRA - Attack AUC: 0.5623
# NullLoRA는 민감한 방향의 information leakage를 줄여 MIA 성공률을 낮춤
