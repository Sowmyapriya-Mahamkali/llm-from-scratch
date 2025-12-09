# FastAPI backend for serving generation endpoint and static UI
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import torch, os
from inference.generate import load_ckpt
from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerModel

app = FastAPI()

# load latest checkpoint if exists
CKPT_PATH = os.path.join('..','checkpoints','latest.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None
tokenizer = None
if os.path.exists(CKPT_PATH):
    vocab, model_state, cfg = load_ckpt(CKPT_PATH, device)
    tokenizer = BPETokenizer(vocab)
    model = TransformerModel(vocab_size=max(tokenizer.vocab.values())+1)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

class GenRequest(BaseModel):
    prompt: str
    length: int = 100
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = None

@app.post('/generate')
async def generate(req: GenRequest):
    if model is None:
        return {'error': 'No model checkpoint found. Train and save a checkpoint at checkpoints/latest.pt'}
    ids = tokenizer.encode(req.prompt)
    import torch
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=req.length, temperature=req.temperature, top_k=req.top_k if req.top_k>0 else None, top_p=req.top_p)
    generated = out[0].cpu().numpy().tolist()
    text = tokenizer.decode(generated)
    return {'text': text}

@app.get('/')
async def root():
    # serve static html
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(os.path.dirname(__file__),'static','index.html'))

if __name__ == '__main__':
    uvicorn.run('web_ui.app:app', host='0.0.0.0', port=8000, reload=True)
