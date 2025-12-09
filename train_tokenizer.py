# train_tokenizer.py
import os
import tokenizer.bpe_tokenizer as bt

# Create tokenizer instance
tokenizer = bt.BPETokenizer()

# Sample training texts (you can add more or read from a dataset)
texts = [
    "Hello world",
    "This is a test",
    "Another example",
    "Machine learning is fun",
    "We are building an LLM from scratch",
    "Natural language processing is interesting"
]

# Train the tokenizer
tokenizer.train(texts)

# Make sure the tokenizer folder exists
os.makedirs("tokenizer", exist_ok=True)

# Save the vocab file
vocab_path = "tokenizer/vocab.json"
tokenizer.save(vocab_path)
print(f"Tokenizer trained and saved at {vocab_path}")

# Test encode/decode
tokens = tokenizer.encode("Hello world")
print("Tokens:", tokens)
decoded = tokenizer.decode(tokens)
print("Decoded text:", decoded)
