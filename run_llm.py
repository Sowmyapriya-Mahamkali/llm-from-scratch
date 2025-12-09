import os
import torch
from tokenizer import bpe_tokenizer as bt
from model.transformer import TransformerModel  # correct model import

# ----------------------------
# Paths
# ----------------------------
VOCAB_PATH = "tokenizer/vocab.json"
CHECKPOINT_PATH = "checkpoints/latest.pt"

# ----------------------------
# Load tokenizer
# ----------------------------
tokenizer = bt.BPETokenizer()
if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(f"Tokenizer vocab not found at {VOCAB_PATH}. Train tokenizer first.")
tokenizer.load(VOCAB_PATH)
print("Tokenizer loaded.")

# ----------------------------
# Instantiate model
# ----------------------------
model = TransformerModel(
    vocab_size=len(tokenizer.vocab),
    d_model=512,       # match your trainer.py config
    n_layers=6,
    n_heads=8,
    d_ff=2048,
    max_len=128,
    dropout=0.1
)

# ----------------------------
# Load checkpoint
# ----------------------------
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Train model first.")

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")  # change to 'cuda' if GPU
model.load_state_dict(checkpoint['model_state'])
model.eval()
print("Model loaded. LLM ready!")

# ----------------------------
# Generation loop
# ----------------------------
print("Type a prompt and press Enter. Type 'exit' or 'quit' to stop.")
while True:
    prompt = input("Enter prompt: ").strip()
    if prompt.lower() in ["exit", "quit"]:
        break

    # Encode prompt
    tokens = tokenizer.encode(prompt)

    # Convert tokens to torch tensor
    input_tensor = torch.tensor([tokens])

    # Forward pass
    with torch.no_grad():
        output_logits = model(input_tensor)
        output_tokens = torch.argmax(output_logits, dim=-1).squeeze().tolist()

    # Decode output tokens
    generated_text = tokenizer.decode(output_tokens)
    print("Generated:", generated_text)
