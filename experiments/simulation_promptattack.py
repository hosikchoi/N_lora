#1. 환경 설정 및 모델 불러오기
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt

model_name = "gpt2"  # 또는 LLaMA / mistral 등
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
#2. 멤버 / 논멤버 Prompt 정의
# 학습에 포함된 것으로 가정할 멤버 prompt
member_prompts = [
    "The capital of France is",
    "The chemical formula for water is",
    "In 1492, Columbus sailed the",
]

# 처음 보는 논멤버 prompt
nonmember_prompts = [
    "The price of Bitcoin in 2025 might be",
    "A theory of quantum consciousness suggests",
    "Zebras in space can survive if",
]
import torch.nn.functional as F

#3. Token-level Output Entropy 계산
def compute_output_entropy(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]  # remove last token for shift
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [batch, seq_len-1]
        mean_entropy = entropy.mean().item()
    return mean_entropy

#4. 모든 Prompt에 대해 Entropy 수집
entropy_mem = [compute_output_entropy(p, model, tokenizer) for p in member_prompts]
entropy_non = [compute_output_entropy(p, model, tokenizer) for p in nonmember_prompts]

print("Avg Entropy (Member):", np.mean(entropy_mem))
print("Avg Entropy (Non-member):", np.mean(entropy_non))

#5. MIA 판단 및 ROC AUC 계산
from sklearn.metrics import roc_auc_score

entropies = np.array(entropy_mem + entropy_non)
labels = np.array([1]*len(entropy_mem) + [0]*len(entropy_non))  # 1 = member

auc = roc_auc_score(labels, -entropies)  # 낮은 entropy일수록 member로 추정
print("🔐 Prompt-level MIA Attack AUC:", auc)

plt.hist(entropy_mem, bins=10, alpha=0.6, label="Member")
plt.hist(entropy_non, bins=10, alpha=0.6, label="Non-member")
plt.axvline(np.mean(entropy_mem), color='blue', linestyle='--')
plt.axvline(np.mean(entropy_non), color='orange', linestyle='--')
plt.legend()
plt.title("Token-level Output Entropy")
plt.xlabel("Entropy")
plt.ylabel("Count")
plt.show()

