import argparse, torch, os
from model.transformer import TransformerModel
from tokenizer.bpe_tokenizer import BPETokenizer

def load_ckpt(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    vocab = ckpt.get('tokenizer', None)
    model_state = ckpt.get('model_state')
    config = ckpt.get('config', None)
    return vocab, model_state, config

def ids_to_text(ids, inv_vocab):
    pieces = [inv_vocab.get(str(i), '') for i in ids]
    return ''.join(pieces).replace('</w>', ' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--prompt', default='Hello')
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab, model_state, cfg = load_ckpt(args.ckpt, device)
    tokenizer = BPETokenizer(vocab)
    inv_vocab = {str(v):k for k,v in tokenizer.vocab.items()}
    model = TransformerModel(vocab_size=max(tokenizer.vocab.values())+1)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    ids = tokenizer.encode(args.prompt)
    import torch
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.length, temperature=args.temperature, top_k=args.top_k if args.top_k>0 else None, top_p=args.top_p)
    generated = out[0].cpu().numpy().tolist()
    print(ids_to_text(generated, inv_vocab))
