from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn.functional import cosine_similarity
from torch import Tensor
import torch.nn.functional as F

app = FastAPI()

model_name_or_path = "/root/paddlejob/workspace/env_run/output/inference/Qwen3-Embedding-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# 定义请求体格式
class EmbeddingRequest(BaseModel):
    texts: list[str]
    max_length: int = 1024
    normalize: bool = True
    # 新增参数：是否返回相似度矩阵（默认返回）
    return_similarity: bool = True

# 定义接口（含余弦相似度计算）
@app.post("/embed")
def embed(request: EmbeddingRequest):
    # 1. 生成嵌入向量
    batch_dict = tokenizer(
        request.texts,
        padding=True,
        truncation=True,
        max_length=request.max_length,
        return_tensors="pt"
    ).to(model.device)
    
    batch_dict.to(model.device)

    with torch.no_grad():
        outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings @ embeddings.T)
    print(scores)
    scores = scores.cpu().numpy().tolist()

    # 3. 构建返回结果
    result = {
    }
    # 若需要返回相似度，添加到结果中
    if request.return_similarity:
        result["similarity_matrix"] = scores
    
    return result

# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)