# Simple educational BPE-style tokenizer.
import re, json
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.inv_vocab = {str(v):k for k,v in self.vocab.items()} if self.vocab else {}

    @staticmethod
    def get_tokens(text):
        return re.findall(r"\S+", text)

    def train(self, corpus, num_merges=500):
        # Very simplified merge-based tokenizer (educational)
        vocab = Counter()
        for line in corpus:
            for token in self.get_tokens(line):
                word = ' '.join(list(token)) + ' </w>'
                vocab[word] += 1
        for i in range(num_merges):
            pairs = Counter()
            for word, freq in vocab.items():
                symbols = word.split()
                for a,b in zip(symbols, symbols[1:]):
                    pairs[(a,b)] += freq
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            bigram = ' '.join(best)
            replacement = ''.join(best)
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = word.replace(bigram, replacement)
                new_vocab[new_word] = freq
            vocab = new_vocab
        # build vocab
        idx = 0
        for word in vocab:
            for piece in word.split():
                if piece not in self.vocab:
                    self.vocab[piece] = idx
                    idx += 1
        self.inv_vocab = {str(v):k for k,v in self.vocab.items()}

    def encode(self, text):
        ids = []
        for token in self.get_tokens(text):
            pieces = list(token) + ['</w>']
            for p in pieces:
                ids.append(self.vocab.get(p, 0))
        return ids

    def decode(self, ids):
        pieces = [self.inv_vocab.get(str(i), '') for i in ids]
        return ''.join(pieces).replace('</w>', ' ')

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            self.inv_vocab = {str(v):k for k,v in self.vocab.items()}
