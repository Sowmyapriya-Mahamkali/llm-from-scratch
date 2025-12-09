import argparse, os, random, yaml, math
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.transformer import TransformerModel
from tokenizer.bpe_tokenizer import BPETokenizer

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
    def __len__(self):
        return max(1, len(self.tokens) - self.block_size)
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1: idx + 1 + self.block_size], dtype=torch.long)
        return x, y

def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)

def save_checkpoint(path, model, tokenizer_vocab, cfg, optimizer=None, scheduler=None, step=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {'model_state': model.state_dict(), 'tokenizer': tokenizer_vocab, 'config': cfg}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler'] = scheduler.state_dict()
    if step is not None:
        payload['step'] = step
    torch.save(payload, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load YAML config safely with UTF-8
    cfg_path = os.path.abspath(args.config)
    print("Loading config from:", cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed',42))
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('device','cuda')=='cuda' else 'cpu')

    # Dataset path
    text_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tiny_shakespeare.txt'))
    print("Looking for dataset at:", text_path)
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Dataset not found at {text_path}")
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train([text], num_merges=300)
    tokens = tokenizer.encode(text)
    vocab_size = max(tokenizer.vocab.values()) + 1

    # Dataset & DataLoader
    block_size = cfg.get('block_size',128)
    dataset = TextDataset(tokens, block_size)
    loader = DataLoader(dataset, batch_size=cfg.get('batch_size',16), shuffle=True)

    # Model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=cfg.get('d_model',512),
        n_layers=cfg.get('n_layers',6),
        n_heads=cfg.get('n_heads',8),
        d_ff=cfg.get('d_ff',2048),
        max_len=cfg.get('block_size',128),
        dropout=cfg.get('dropout',0.1)
    )
    model.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('learning_rate',0.0003))
    total_steps = math.ceil(len(loader) * cfg.get('epochs',1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1,total_steps))
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Checkpoints
    checkpoint_dir = cfg.get('checkpoint_dir','checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if provided
    step = 0
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer' in checkpoint and 'scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        step = checkpoint.get('step', 0)

    # Training loop
    for epoch in range(cfg.get('epochs',1)):
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(xb)
                loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            pbar.set_postfix({'loss': float(loss), 'lr': scheduler.get_last_lr()[0], 'step': step})
            if step % cfg.get('save_every',500) == 0:
                save_checkpoint(os.path.join(checkpoint_dir, f'ckpt_{step}.pt'), model, tokenizer.vocab, cfg, optimizer, scheduler, step)

    # final save
    save_checkpoint(os.path.join(checkpoint_dir, 'latest.pt'), model, tokenizer.vocab, cfg, optimizer, scheduler, step)
    print("Training complete! Checkpoints saved at:", checkpoint_dir)

if __name__ == '__main__':
    main()
