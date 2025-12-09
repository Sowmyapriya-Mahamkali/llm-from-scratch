import os
import torch
from tokenizer import bpe_tokenizer as bt
from training.model import LLM  # adjust import if your model is in a different file

# Path to vocab file
VOCAB_PATH = "tokenizer/vocab.json"

# Load or train tokenizer
tokenizer = bt.BPETokenizer()
if not os.path.exists(VOCAB_PATH):
    print("vocab.json not found. Training tokenizer...")
    tokenizer.train("data/training_text.txt")  # put your dataset path here
    tokenizer.save(VOCAB_PATH)
    print(f"Tokenizer trained and saved at {VOCAB_PATH}")

tokenizer.load(VOCAB_PATH)
print("Tokenizer loaded.")

# Load your trained LLM model
model = LLM()  # instantiate your model
model.load_state_dict(torch.load("checkpoints/latest.pt"))
 # adjust path
model.eval()
print("Model loaded. LLM ready!")

# Generation loop
print("Type a prompt and press Enter. Type 'exit' or 'quit' to stop.")

while True:
    prompt = input("Enter prompt: ").strip()
    if prompt.lower() in ["exit", "quit"]:
        break

    # Encode prompt
    tokens = tokenizer.encode(prompt)

    # Convert tokens to torch tensor
    input_tensor = torch.tensor([tokens])

    # Generate output tokens (example: simple forward pass)
    with torch.no_grad():
        output_logits = model(input_tensor)  # shape: [1, seq_len, vocab_size]
        output_tokens = torch.argmax(output_logits, dim=-1).squeeze().tolist()

    # Decode tokens to text
    generated_text = tokenizer.decode(output_tokens)
    print("Generated:", generated_text)
