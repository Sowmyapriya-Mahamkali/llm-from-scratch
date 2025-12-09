# LLM From Scratch — Full Project (Enhanced)

This is a full, runnable mini-GPT style repository implemented in PyTorch with:

- Tokenizer (toy BPE) and optional HuggingFace `tokenizers` integration
- Modular transformer implementation with multi-head causal attention
- Improved generation (top-k and nucleus/top-p)
- Training loop with mixed precision (`torch.amp`), LR scheduler, checkpointing
- Simple FastAPI backend exposing `/generate` endpoint
- Minimal browser chat UI (static HTML + JS) that calls the backend
- Clear project structure

---

## Quick Start (Locally)

```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Place a training text file at data/tiny_shakespeare.txt
python training/trainer.py --config config.yaml

# Start API server
python api/server.py

