import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm


MODEL_NAME = "klue/roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 128

sts = load_dataset("mteb/KorSTS")

# 데이터 확인
print(sts["train"][2])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def collate_batch(batch):
    s1 = [x["sentence1"] for x in batch]
    s2 = [x["sentence2"] for x in batch]
    labels = [x["score"] / 5.0 for x in batch]  # 0~1로 정규화

    e1 = tokenizer(s1, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    e2 = tokenizer(s2, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")


    e1.pop("token_type_ids", None)
    e2.pop("token_type_ids", None)

    labels = torch.tensor(labels, dtype=torch.float)
    return e1, e2, labels

train_loader = DataLoader(sts["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

class SRoBERTa(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pooling="mean"):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mean":
            emb = self.mean_pooling(outputs, attention_mask)
        else:
            emb = outputs.last_hidden_state[:, 0]
        return nn.functional.normalize(emb, p=2, dim=1)  # L2 정규화

model = SRoBERTa().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for e1, e2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        e1 = {k: v.to(DEVICE) for k, v in e1.items()}
        e2 = {k: v.to(DEVICE) for k, v in e2.items()}
        labels = labels.to(DEVICE)

        v1 = model(**e1)
        v2 = model(**e2)

        cos_sim = nn.functional.cosine_similarity(v1, v2)
        loss = loss_fn(cos_sim, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

model.eval()
test_pairs = [
    ("오늘 날씨가 참 좋다.", "하늘이 맑고 기분이 좋다."),
    ("고양이가 소파에 앉아 있다.", "강아지가 달리고 있다."),
]

with torch.no_grad():
    for s1, s2 in test_pairs:
        t1 = tokenizer(s1, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        t2 = tokenizer(s2, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        v1 = model(**t1)
        v2 = model(**t2)
        sim = nn.functional.cosine_similarity(v1, v2).item()
        print(f"🔹 [{s1}] vs [{s2}] → similarity = {sim:.4f}")