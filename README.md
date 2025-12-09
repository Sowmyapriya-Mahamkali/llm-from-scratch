# LLM From Scratch — Full Project (Enhanced)

This is a full, runnable mini-GPT style repository implemented in PyTorch with:
- tokenizer (toy BPE) and optional HuggingFace `tokenizers` integration
- modular transformer implementation with multi-head causal attention
- improved generation (top-k and nucleus/top-p)
- training loop with mixed precision (torch.amp), LR scheduler, checkpointing (saves config)
- simple FastAPI backend exposing `/generate` endpoint
- minimal browser chat UI (static HTML + JS) that calls the backend
- clear project structure

## Quick start (locally)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# Place a training text file at data/tiny_shakespeare.txt
python training/trainer.py --config config.yaml
# Start API server:
uvicorn web_ui.app:app --reload --host 0.0.0.0 --port 8000
# Open browser at http://localhost:8000
```
